"""Shared evaluation code for KalmanFilter, KalmanNet and SSMPhysics on RADIATE test tracks.

Run directly to evaluate all three models on the clean test set:
    python run.py --ssm_physics_ckpt checkpoints/ssm_physics_10frames/best_model.pth \
                  --knet_ckpt checkpoints/knet/best_model.pth --use_cuda
"""

import os
import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.ssm_rt_physics import SSMPhysicsTracker
from model.kalman import KalmanFilter
from model.kalmannet import KalmanNetNN
from model.system_model import TrackingSystemModel
from dataset.radiate_dataset import RadiateDataset
from utils.utils import (
    count_parameters, compute_pos_mse, compute_vel_mse, compute_nstep_mse, compute_mse, mse_to_db
)

MODELS = ['KalmanFilter', 'KalmanNet', 'SSMPhysics']


class Args:
    """Simple attribute container (used to build KalmanNet)."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def evaluate_model(model, test_loader, device, model_name, state_dim=7):
    """
    Evaluate a model on the test set.

    KalmanFilter and KalmanNet are initialised with the ground-truth first state and run frame by frame.
    SSMPhysics processes the whole sequence in one pass; its physics prior at step t is built from the
    ground-truth state at t-1 (teacher forcing, as in training).

    Inference time: sequence-level ms. SSMPhysics: first sequence; KalmanNet: mean per-step time on the
    first sequence x sequence length; KalmanFilter: mean over the first 10 sequences.

    Returns:
        dict of metrics (MSE values in dB)
    """
    model.eval()

    all_pred_states = []
    all_target_states = []
    all_seq_lens = []
    inference_times = []

    print(f"\nEvaluating {model_name}...")

    # Warmup
    with torch.no_grad():
        if model_name == 'KalmanNet':
            obs_dim = test_loader.dataset[0]['observation'].shape[-1]
            model.batch_size = 1
            model.InitSequence(torch.zeros(1, state_dim, 1).to(device), 10)
            model.init_hidden_KNet()
            for _ in range(5):
                _ = model(torch.randn(1, obs_dim, 1).to(device))
            _sync(device)
        elif model_name == 'SSMPhysics':
            sample_obs = torch.randn(1, 10, test_loader.dataset[0]['observation'].shape[-1]).to(device)
            for _ in range(5):
                _ = model(sample_obs)
            _sync(device)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            obs = batch['observation'].to(device)  # [batch, seq_len, obs_dim]
            state = batch['state'].to(device)  # [batch, seq_len, state_dim]
            seq_len = batch['seq_len']

            for b in range(obs.shape[0]):
                seq_len_b = seq_len[b].item()
                obs_seq = obs[b, :seq_len_b, :]  # [seq_len, obs_dim]
                state_seq = state[b, :seq_len_b, :]  # [seq_len, state_dim]
                first = batch_idx == 0 and b == 0

                if model_name == 'SSMPhysics':
                    obs_batch = obs_seq.unsqueeze(0)  # [1, seq_len, obs_dim]
                    init_state = state_seq[0:1, :].squeeze(0)  # [state_dim]
                    prev_state = torch.cat([
                        torch.zeros(1, 1, state_dim, device=device),
                        state_seq[:-1, :].unsqueeze(0)
                    ], dim=1)  # [1, T, state_dim]

                    if first:
                        _sync(device)
                        start = time.time()
                    pred_state_seq = model(obs_batch, init_state=init_state, prev_state=prev_state)
                    if first:
                        _sync(device)
                        inference_times.append((time.time() - start) * 1000)
                    pred_state_seq = pred_state_seq.squeeze(0)  # [seq_len, state_dim]

                elif model_name == 'KalmanNet':
                    m1x_0 = state_seq[0:1, :].transpose(0, 1).unsqueeze(0)  # [1, state_dim, 1]
                    model.batch_size = 1
                    model.InitSequence(m1x_0, seq_len_b)
                    model.init_hidden_KNet()

                    pred_states = []
                    per_step_times = []
                    for t in range(seq_len_b):
                        y_t = obs_seq[t:t+1, :].transpose(0, 1).unsqueeze(0)  # [1, obs_dim, 1]
                        if first:
                            _sync(device)
                            start = time.time()
                        pred_state_t = model(y_t)  # [1, state_dim, 1]
                        if first:
                            _sync(device)
                            per_step_times.append((time.time() - start) * 1000)
                        pred_states.append(pred_state_t.squeeze(0).squeeze(-1))
                    if first:
                        inference_times.append(np.mean(per_step_times) * seq_len_b)
                    pred_state_seq = torch.stack(pred_states, dim=0)  # [seq_len, state_dim]

                elif model_name == 'KalmanFilter':
                    obs_batch = obs_seq.unsqueeze(0)  # [1, seq_len, obs_dim]
                    m1x_0 = state_seq[0:1, :].transpose(0, 1).unsqueeze(0)  # [1, state_dim, 1]
                    timed = len(inference_times) < 10
                    if timed:
                        _sync(device)
                        start = time.time()
                    pred_state_seq = model(obs_batch, m1x_0=m1x_0)[0]  # [seq_len, state_dim]
                    if timed:
                        _sync(device)
                        inference_times.append((time.time() - start) * 1000)

                else:
                    raise ValueError(f"Unknown model: {model_name}")

                if torch.isnan(pred_state_seq).any() or torch.isinf(pred_state_seq).any():
                    print(f"Warning: {model_name} produced NaN/inf in sequence {batch_idx * obs.shape[0] + b}; set to 0")
                    pred_state_seq = torch.nan_to_num(pred_state_seq, nan=0.0, posinf=0.0, neginf=0.0)

                all_pred_states.append(pred_state_seq.cpu())
                all_target_states.append(state_seq.cpu())
                all_seq_lens.append(seq_len_b)

    params = count_parameters(model)
    avg_inference_time = np.mean(inference_times) if inference_times else 0.0

    pos_mses, vel_mses = [], []
    nstep_pos_mses, nstep_vel_mses = [], []
    nstep_mses = []      # MSE over the last 5 frames (all state dims)
    total_mses = []      # MSE over the full sequence (all state dims)
    sliding_5_mses = []  # MSE averaged over all 5-frame windows

    for pred_seq, target_seq, seq_len_b in zip(all_pred_states, all_target_states, all_seq_lens):
        pred_seq = pred_seq[:seq_len_b]
        target_seq = target_seq[:seq_len_b]

        pos_mse = compute_pos_mse(pred_seq, target_seq)
        vel_mse = compute_vel_mse(pred_seq, target_seq)
        total_mse = compute_mse(pred_seq, target_seq)

        if seq_len_b >= 5:
            pred_nstep = pred_seq[-5:, :]
            target_nstep = target_seq[-5:, :]
            nstep_pos_mse = compute_pos_mse(pred_nstep, target_nstep)
            nstep_vel_mse = compute_vel_mse(pred_nstep, target_nstep)
            nstep_mse = compute_nstep_mse(pred_nstep, target_nstep, n=5)
            window_mses = [compute_mse(pred_seq[end-4:end+1, :], target_seq[end-4:end+1, :])
                           for end in range(4, seq_len_b)]
            sliding_5_mse = float(np.mean(window_mses))
        else:
            nstep_pos_mse = pos_mse
            nstep_vel_mse = vel_mse
            nstep_mse = total_mse
            sliding_5_mse = total_mse

        pos_mses.append(pos_mse)
        vel_mses.append(vel_mse)
        nstep_pos_mses.append(nstep_pos_mse)
        nstep_vel_mses.append(nstep_vel_mse)
        nstep_mses.append(nstep_mse)
        total_mses.append(total_mse)
        sliding_5_mses.append(sliding_5_mse)

    # Rough FLOPs estimate: 2 ops per trainable parameter per time step
    num_sequences = len(all_seq_lens)
    avg_seq_len = float(np.mean(all_seq_lens)) if num_sequences > 0 else 0.0
    flops_per_sequence_est = 2.0 * params * avg_seq_len

    return {
        'params': params,
        'inference_time_ms': avg_inference_time,
        'pos_mse_db': mse_to_db(np.mean(pos_mses)),
        'vel_mse_db': mse_to_db(np.mean(vel_mses)),
        'nstep_pos_mse_db': mse_to_db(np.mean(nstep_pos_mses)),
        'nstep_vel_mse_db': mse_to_db(np.mean(nstep_vel_mses)),
        'nstep_mse_db': mse_to_db(np.mean(nstep_mses)),        # "5-step MSE" (last 5 frames)
        'total_mse_db': mse_to_db(np.mean(total_mses)),        # "MSE" (full sequence)
        'sliding_5_mse_db': mse_to_db(np.mean(sliding_5_mses)),
        'flops_per_sequence_est': flops_per_sequence_est,
        'total_flops_est': flops_per_sequence_est * num_sequences,
    }


def load_model(model_type, checkpoint_path, args, device):
    """Build a model and load its checkpoint (KalmanFilter has none)."""
    sys_model = TrackingSystemModel(state_dim=args.state_dim, obs_dim=args.obs_dim, dt=1.0, device=device)

    if model_type == 'SSMPhysics':
        model = SSMPhysicsTracker(
            obs_dim=args.obs_dim,
            state_dim=args.state_dim,
            d_model=args.d_model,
            d_state=args.d_state,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_kalman_features=True,
            use_system_model=True,
            system_model=sys_model,
            device=device,
            ssm_type=args.ssm_type,
        ).to(device)
        print(f"SSMPhysics config: {model.get_config()}")

    elif model_type == 'KalmanFilter':
        model = KalmanFilter(F=sys_model.F, H=sys_model.H, Q=sys_model.Q, R=sys_model.R, device=device).to(device)

    elif model_type == 'KalmanNet':
        knet_args = Args(
            use_cuda=device.type == 'cuda',
            n_batch=1,
            in_mult_KNet=args.in_mult_KNet,
            out_mult_KNet=args.out_mult_KNet,
        )
        model = KalmanNetNN()
        model.NNBuild(sys_model, knet_args)
        model = model.to(device)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")

    return model


def add_common_args(parser):
    """Arguments shared by all evaluation scripts."""
    parser.add_argument('--test_dataset_root', type=str, default='./data/radiate/test',
                        help='Root directory of the RADIATE test split')
    parser.add_argument('--ssm_physics_ckpt', type=str, default='./checkpoints/ssm_physics_10frames/best_model.pth')
    parser.add_argument('--knet_ckpt', type=str, default='./checkpoints/knet/best_model.pth')
    parser.add_argument('--models', type=str, nargs='+', default=MODELS, choices=MODELS)
    parser.add_argument('--seq_len', type=int, default=10, help='Track length (tracks are split into windows of this length)')
    parser.add_argument('--obs_dim', type=int, default=5)
    parser.add_argument('--state_dim', type=int, default=7)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--d_state', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--in_mult_KNet', type=int, default=5)
    parser.add_argument('--out_mult_KNet', type=int, default=40)
    parser.add_argument('--ssm_type', type=str, default='mamba1_cuda', choices=['mamba1_cuda', 'mamba1'],
                        help='mamba1_cuda: mamba_ssm CUDA kernels; mamba1: pure-PyTorch fallback (same weights)')
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA')
    return parser


def checkpoint_for(model_name, args):
    return {'SSMPhysics': args.ssm_physics_ckpt, 'KalmanNet': args.knet_ckpt, 'KalmanFilter': None}[model_name]


def make_device(args):
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu' and args.ssm_type == 'mamba1_cuda':
        print("CPU run: switching SSMPhysics to the pure-PyTorch core (ssm_type=mamba1)")
        args.ssm_type = 'mamba1'
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == 'cuda' else ''))
    return device


def main():
    parser = add_common_args(argparse.ArgumentParser(description='Evaluate KF, KalmanNet and SSMPhysics on the clean test set'))
    args = parser.parse_args()
    device = make_device(args)

    test_dataset = RadiateDataset(args.test_dataset_root, min_seq_len=args.seq_len, max_seq_len=args.seq_len,
                                  obs_dim=args.obs_dim, state_dim=args.state_dim)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f'Test dataset size: {len(test_dataset)} sequences')

    results = []
    for model_name in args.models:
        model = load_model(model_name, checkpoint_for(model_name, args), args, device)
        results.append((model_name, evaluate_model(model, test_loader, device, model_name, args.state_dim)))

    print(f"\nClean test set, {args.seq_len}-frame tracks ({len(test_dataset)} sequences)")
    print(f"{'Model':<14} {'Params':>10} {'Time (ms)':>10} {'MSE (dB)':>10} {'5-step MSE (dB)':>16} {'Pos MSE (dB)':>13}")
    for model_name, m in results:
        print(f"{model_name:<14} {m['params']:>10,} {m['inference_time_ms']:>10.2f} "
              f"{m['total_mse_db']:>10.2f} {m['nstep_mse_db']:>16.2f} {m['pos_mse_db']:>13.2f}")


if __name__ == '__main__':
    main()
