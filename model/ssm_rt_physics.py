"""Physics-informed selective SSM Kalman filter (SSMPhysics) for RADIATE bounding-box tracking.

ssm_type selects the implementation of the selective SSM (Mamba-1) layers:
  mamba1_cuda : mamba_ssm.Mamba (CUDA/Triton kernels)
  mamba1      : pure-PyTorch Mamba-1 (same parameters and math; loads the same checkpoints)
"""

import torch
import torch.nn as nn

from model.ssm_blocks import SSMBlock


def _cuda_mamba_cls():
    try:
        from mamba_ssm import Mamba
    except Exception as e:
        raise ImportError("Could not import mamba_ssm.Mamba (CUDA build). Use ssm_type='mamba1' for the "
                          f"pure-PyTorch equivalent. Original error: {e}")
    return Mamba


class SSMPhysicsTracker(nn.Module):
    """
    Selective SSM with a constant-velocity physics prior:
    - Uses the system model (F, H) to compute x_prior_t, y_prior_t and the innovation
    - The SSM encoder predicts a Kalman-gain matrix K_t and a residual delta_t
    - Output: x_hat_t = x_prior_t + K_t * innovation_t + 0.1 * delta_t   (Eq. 8 of the paper, lambda = 0.1)

    Inputs:
        obs: [B, T, obs_dim]
        init_state: [B, state_dim] (optional; used only when prev_state is None)
        prev_state: [B, T, state_dim] (state used to build the prior at each step)
    Output:
        x_hat: [B, T, state_dim]
    """

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        d_model: int = 256,
        d_state: int = 32,
        num_layers: int = 6,
        dropout: float = 0.1,
        use_kalman_features: bool = True,
        use_system_model: bool = False,
        system_model=None,
        device: str = "cpu",
        ssm_type: str = "mamba1_cuda",
        expand: int = 2,
        d_conv: int = 4,
        feat_clamp: float = 1000.0,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.d_model = d_model
        self.d_state = d_state
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.use_kalman_features = use_kalman_features
        self.use_system_model = use_system_model
        self.system_model = system_model
        self.device = device
        self.ssm_type = ssm_type
        self.expand = expand
        self.d_conv = d_conv
        self.feat_clamp = feat_clamp

        if self.use_system_model and self.system_model is None:
            raise ValueError("use_system_model=True but system_model is None")

        # Feature dimension calculation
        feature_dim = obs_dim
        if self.use_kalman_features:
            feature_dim += obs_dim  # obs_diff
        if self.use_system_model:
            feature_dim += obs_dim  # innovation
            if self.use_kalman_features:
                feature_dim += obs_dim  # obs_diff_norm
                feature_dim += state_dim  # state_prior_diff_norm
            feature_dim += state_dim  # x_prior
        self.feature_dim = feature_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

        # Selective SSM layers
        if ssm_type == "mamba1_cuda":
            Mamba = _cuda_mamba_cls()
            self.mamba_layers = nn.ModuleList([
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, device=device)
                for _ in range(num_layers)
            ])
        elif ssm_type == "mamba1":
            self.mamba_layers = nn.ModuleList([
                SSMBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, device=device)
                for _ in range(num_layers)
            ])
        else:
            raise ValueError(f"unknown ssm_type {ssm_type}")
        self.norm_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # Kalman gain head
        self.innovation_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, obs_dim * state_dim)  # Kalman gain matrix elements
        )

        # Residual head
        self.residual_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, state_dim)
        )

        self._init_weights()
        self.to(device)

    def get_config(self):
        return dict(
            obs_dim=self.obs_dim, state_dim=self.state_dim, d_model=self.d_model, d_state=self.d_state,
            num_layers=self.num_layers, dropout=self.dropout_p, use_kalman_features=self.use_kalman_features,
            use_system_model=self.use_system_model, ssm_type=self.ssm_type, expand=self.expand,
            d_conv=self.d_conv, feature_dim=self.feature_dim, feat_clamp=self.feat_clamp,
        )

    def _init_weights(self):
        """Initialize weights for better convergence."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if getattr(getattr(module, "bias", None), "_no_reinit", False):
                    continue  # keep the special dt bias init of SSM blocks
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0.0)
                nn.init.constant_(module.weight, 1.0)

    # ------------------------------------------------------------------ physics prior
    def _priors_from_prev_state(self, prev_state):
        """x_prior_t = F prev_state_t, y_prior_t = H x_prior_t, vectorized over time.
        prev_state: [B, T, state_dim] -> ([B, T, state_dim], [B, T, obs_dim])"""
        Fm = self.system_model.F.to(prev_state.dtype)
        Hm = self.system_model.H.to(prev_state.dtype)
        x_prior_seq = prev_state @ Fm.T
        y_prior_seq = x_prior_seq @ Hm.T
        return x_prior_seq, y_prior_seq

    def _compute_physics_priors(self, obs, init_state):
        """Free-running prior: x_prior_t = F^t init_state (no measurement update).
        Returns x_prior_seq [B, T, state_dim], y_prior_seq [B, T, obs_dim]."""
        B, T, _ = obs.shape
        Fm = self.system_model.F.to(init_state.dtype)
        Hm = self.system_model.H.to(init_state.dtype)
        F_pows = [torch.eye(self.state_dim, device=init_state.device, dtype=init_state.dtype)]
        for _ in range(1, T):
            F_pows.append(Fm @ F_pows[-1])
        F_pows = torch.stack(F_pows, dim=0)  # [T, m, m]
        x_prior_seq = torch.einsum("tij,bj->bti", F_pows, init_state)
        y_prior_seq = x_prior_seq @ Hm.T
        x_prior_seq = torch.nan_to_num(torch.clamp(x_prior_seq, min=-1e6, max=1e6), nan=0.0, posinf=1e6, neginf=-1e6)
        y_prior_seq = torch.nan_to_num(torch.clamp(y_prior_seq, min=-1e6, max=1e6), nan=0.0, posinf=1e6, neginf=-1e6)
        return x_prior_seq, y_prior_seq

    # ------------------------------------------------------------------ forward
    def forward(self, obs: torch.Tensor, init_state: torch.Tensor = None, prev_state: torch.Tensor = None):
        """
        Args:
            obs: [B, T, obs_dim]
            init_state: [B, state_dim] (initial state for the free-running prior; used only if prev_state is None)
            prev_state: [B, T, state_dim] (state used to build the prior at each step)
        Returns:
            x_hat: [B, T, state_dim]
        """
        B, T, _ = obs.shape
        device = obs.device

        feats = [obs]

        if self.use_kalman_features:
            obs_diff = obs - torch.cat([
                torch.zeros(B, 1, self.obs_dim, device=device, dtype=obs.dtype),
                obs[:, :-1, :]
            ], dim=1)
            feats.append(obs_diff)

        if self.use_system_model:
            if prev_state is not None:
                x_prior_seq, y_prior_seq = self._priors_from_prev_state(prev_state)
            else:
                if init_state is None:
                    init_state = torch.zeros(B, self.state_dim, device=device, dtype=obs.dtype)
                x_prior_seq, y_prior_seq = self._compute_physics_priors(obs, init_state)

            innovation = obs - y_prior_seq

            if self.use_kalman_features:
                state_prior_diff = x_prior_seq - torch.cat([
                    torch.zeros(B, 1, self.state_dim, device=device, dtype=obs.dtype),
                    x_prior_seq[:, :-1, :]
                ], dim=1)
                innovation_norm = torch.nn.functional.normalize(innovation, p=2, dim=-1, eps=1e-12)
                obs_diff_norm = torch.nn.functional.normalize(obs_diff, p=2, dim=-1, eps=1e-12)
                state_prior_diff_norm = torch.nn.functional.normalize(state_prior_diff, p=2, dim=-1, eps=1e-12)
                feats.append(innovation_norm)
                feats.append(obs_diff_norm)
                feats.append(state_prior_diff_norm)
            else:
                feats.append(innovation)

            feats.append(x_prior_seq)

        feats = torch.cat(feats, dim=-1)  # [B, T, feature_dim]
        feats = torch.clamp(feats, min=-self.feat_clamp, max=self.feat_clamp)
        x = self.input_proj(feats)

        for i, (mamba_layer, norm_layer, ffn) in enumerate(zip(self.mamba_layers, self.norm_layers, self.ffn_layers)):
            residual = x
            x = mamba_layer(x)
            x = norm_layer(x)
            x_ffn = ffn(x)
            if i > 0:
                x = x + x_ffn + residual
            else:
                x = x + x_ffn  # NOTE: layer 0 has no skip connection (as in the released checkpoint)
            if i < len(self.mamba_layers) - 1:
                x = self.dropout(x)

        if self.use_system_model:
            gain = self.innovation_proj(x).view(B, T, self.state_dim, self.obs_dim)  # K_t
            innovation_update = torch.einsum("btij,btj->bti", gain, innovation)
            x_hat = x_prior_seq + innovation_update
            x_hat = x_hat + 0.1 * self.residual_proj(x)
        else:
            x_hat = self.residual_proj(x)

        x_hat = torch.nan_to_num(torch.clamp(x_hat, min=-1e6, max=1e6), nan=0.0, posinf=1e6, neginf=-1e6)
        return x_hat
