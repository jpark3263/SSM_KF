"""System Model for tracking with KalmanNet."""

import torch


class TrackingSystemModel:
    """
    System Model for object tracking.
    
    State: [cx, cy, vx, vy, w, h, angle] (7D)
    Observation: [cx, cy, w, h, angle] (5D)
    
    Uses Constant Velocity (CV) model.
    """
    
    def __init__(self, state_dim=7, obs_dim=5, dt=1.0, device='cpu'):
        """
        Initialize system model.
        
        Args:
            state_dim: State dimension (6 or 7)
            obs_dim: Observation dimension (4 or 5)
            dt: Time step
            device: Device
        """
        self.m = state_dim
        self.n = obs_dim
        self.dt = dt
        self.device = device
        
        # State transition matrix F (Constant Velocity model)
        # State: [cx, cy, vx, vy, w, h, angle]
        self.F = self._build_F(state_dim, dt).to(device)
        
        # Observation matrix H
        # Observation: [cx, cy, w, h, angle] (velocity not observed)
        self.H = self._build_H(state_dim, obs_dim).to(device)
        
        # Process noise Q
        q_pos = 0.1  # Position noise
        q_vel = 0.5  # Velocity noise
        q_size = 0.01  # Size noise
        q_angle = 0.1  # Angle noise
        self.Q = self._build_Q(state_dim, q_pos, q_vel, q_size, q_angle).to(device)
        
        # Observation noise R
        r_pos = 1.0  # Position observation noise
        r_size = 0.1  # Size observation noise
        r_angle = 0.5  # Angle observation noise
        self.R = self._build_R(obs_dim, r_pos, r_size, r_angle).to(device)
        
        # Priors
        self.prior_Q = torch.eye(self.m).to(device)
        self.prior_Sigma = torch.zeros((self.m, self.m)).to(device)
        self.prior_S = torch.eye(self.n).to(device)
        
        self.T = 100  # Default sequence length
        self.T_test = 100
    
    def _build_F(self, state_dim, dt):
        """Build state transition matrix F."""
        if state_dim == 7:
            # [cx, cy, vx, vy, w, h, angle]
            F = torch.eye(7)
            F[0, 2] = dt  # cx += vx * dt
            F[1, 3] = dt  # cy += vy * dt
            # w, h, angle remain constant
        elif state_dim == 6:
            # [cx, cy, vx, vy, w, h]
            F = torch.eye(6)
            F[0, 2] = dt
            F[1, 3] = dt
        else:
            raise ValueError(f"Unsupported state_dim: {state_dim}")
        return F
    
    def _build_H(self, state_dim, obs_dim):
        """Build observation matrix H."""
        if state_dim == 7 and obs_dim == 5:
            # State: [cx, cy, vx, vy, w, h, angle]
            # Obs: [cx, cy, w, h, angle]
            H = torch.zeros(5, 7)
            H[0, 0] = 1  # cx -> cx
            H[1, 1] = 1  # cy -> cy
            H[2, 4] = 1  # w -> w
            H[3, 5] = 1  # h -> h
            H[4, 6] = 1  # angle -> angle
        elif state_dim == 6 and obs_dim == 4:
            # State: [cx, cy, vx, vy, w, h]
            # Obs: [cx, cy, w, h]
            H = torch.zeros(4, 6)
            H[0, 0] = 1  # cx -> cx
            H[1, 1] = 1  # cy -> cy
            H[2, 4] = 1  # w -> w
            H[3, 5] = 1  # h -> h
        else:
            raise ValueError(f"Unsupported combination: state_dim={state_dim}, obs_dim={obs_dim}")
        return H
    
    def _build_Q(self, state_dim, q_pos, q_vel, q_size, q_angle):
        """Build process noise covariance Q."""
        if state_dim == 7:
            Q = torch.diag(torch.tensor([q_pos, q_pos, q_vel, q_vel, q_size, q_size, q_angle]))
        elif state_dim == 6:
            Q = torch.diag(torch.tensor([q_pos, q_pos, q_vel, q_vel, q_size, q_size]))
        else:
            raise ValueError(f"Unsupported state_dim: {state_dim}")
        return Q
    
    def _build_R(self, obs_dim, r_pos, r_size, r_angle):
        """Build observation noise covariance R."""
        if obs_dim == 5:
            R = torch.diag(torch.tensor([r_pos, r_pos, r_size, r_size, r_angle]))
        elif obs_dim == 4:
            R = torch.diag(torch.tensor([r_pos, r_pos, r_size, r_size]))
        else:
            raise ValueError(f"Unsupported obs_dim: {obs_dim}")
        return R
    
    def f(self, x):
        """State evolution function."""
        # x: [batch_size, m, 1]
        batched_F = self.F.view(1, self.m, self.m).expand(x.shape[0], -1, -1)
        return torch.bmm(batched_F, x)
    
    def h(self, x):
        """Observation function."""
        # x: [batch_size, m, 1]
        batched_H = self.H.view(1, self.n, self.m).expand(x.shape[0], -1, -1)
        return torch.bmm(batched_H, x)
