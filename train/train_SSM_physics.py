"""Train the physics-informed selective SSM Kalman filter (SSMPhysics) on RADIATE tracks.

Paper setting (10-frame tracks, 5000 epochs):
    python train/train_SSM_physics.py --dataset_root data/radiate/train --use_cuda
"""

import os
import sys
import argparse
import csv
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.ssm_rt_physics import SSMPhysicsTracker
from model.system_model import TrackingSystemModel
from dataset.radiate_dataset import RadiateDataset
from utils.utils import count_parameters


def masked_mse(pred, target, seq_len):
    """MSE over valid frames only (sequences are right-padded to max_seq_len)."""
    B, T, D = pred.shape
    valid = (torch.arange(T, device=pred.device).view(1, T) < seq_len.view(B, 1)).to(pred.dtype).unsqueeze(-1)
    return (((pred - target) ** 2) * valid).sum() / (valid.sum() * D)


def teacher_forcing_prev_state(state):
    """Prior source at step t = ground-truth state at t-1 (zeros at t=0)."""
    return torch.cat([torch.zeros(state.shape[0], 1, state.shape[2], device=state.device), state[:, :-1, :]], dim=1)


def train(args):
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    if device.type == 'cpu' and args.ssm_type == 'mamba1_cuda':
        args.ssm_type = 'mamba1'
    print(f'Using device: {device}, ssm_type: {args.ssm_type}')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset = RadiateDataset(
        dataset_root=args.dataset_root,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
        obs_dim=args.obs_dim,
        state_dim=args.state_dim,
    )
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    sys_model = TrackingSystemModel(state_dim=args.state_dim, obs_dim=args.obs_dim, dt=1.0, device=device)
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

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate drops to 0.1x when val loss < 10 and to 0.01x when val loss < 1
    current_lr = args.lr
    lr_threshold_1 = 10.0
    lr_threshold_2 = 1.0

    num_params = count_parameters(model)
    print(f'Parameters: {num_params:,}')
    print(f'Train / val sequences: {len(train_dataset)} / {len(val_dataset)}')

    best_val_loss = float('inf')
    best_epoch = 0
    os.makedirs(args.save_dir, exist_ok=True)
    log_csv_path = os.path.join(args.save_dir, 'training_log.csv')
    with open(log_csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'learning_rate', 'epoch_sec'])

    print_every = 50
    for epoch in range(args.epochs):
        should_print = (epoch + 1) % print_every == 0 or epoch == 0 or epoch == args.epochs - 1
        epoch_t0 = time.time()

        # Training (teacher forcing: the prior at t comes from the ground-truth state at t-1)
        model.train()
        train_loss, train_batch_count = 0.0, 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]', disable=not should_print):
            obs = torch.clamp(batch['observation'].to(device), min=-1e6, max=1e6)
            state = torch.clamp(batch['state'].to(device), min=-1e6, max=1e6)

            optimizer.zero_grad()
            pred_state = model(obs, init_state=state[:, 0, :], prev_state=teacher_forcing_prev_state(state))
            if not torch.isfinite(pred_state).all():
                continue  # skip numerically unstable batch

            loss = masked_mse(pred_state, state, batch['seq_len'].to(device))
            if not torch.isfinite(loss):
                continue
            loss.backward()

            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if not torch.isfinite(grad_norm):
                    optimizer.zero_grad()
                    continue
            optimizer.step()

            train_loss += loss.item()
            train_batch_count += 1
        train_loss = train_loss / train_batch_count if train_batch_count > 0 else float('inf')

        # Validation
        model.eval()
        val_loss, val_batch_count = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                obs = batch['observation'].to(device)
                state = batch['state'].to(device)
                pred_state = model(obs, init_state=torch.clamp(state[:, 0, :], min=-1e6, max=1e6),
                                   prev_state=teacher_forcing_prev_state(state))
                if not torch.isfinite(pred_state).all():
                    continue
                loss = masked_mse(pred_state, state, batch['seq_len'].to(device))
                if not torch.isfinite(loss):
                    continue
                val_loss += loss.item()
                val_batch_count += 1
        val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')

        if val_loss < lr_threshold_2 and current_lr > args.lr * 0.01:
            current_lr = args.lr * 0.01
        elif val_loss < lr_threshold_1 and current_lr > args.lr * 0.1:
            current_lr = args.lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        if should_print:
            print(f'Epoch {epoch+1}/{args.epochs}: train {train_loss:.4f}, val {val_loss:.4f}, '
                  f'lr {current_lr:.6f}, {time.time()-epoch_t0:.1f}s')
        with open(log_csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, train_loss, val_loss, current_lr, round(time.time() - epoch_t0, 2)])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': model.get_config(),
                'train_args': vars(args),
            }, os.path.join(args.save_dir, 'best_model.pth'))

    summary = (f"Model: SSMPhysicsTracker ({num_params:,} parameters)\n"
               f"Finished: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
               f"Train / val sequences: {len(train_dataset)} / {len(val_dataset)}\n"
               f"Best val loss: {best_val_loss:.6f} (epoch {best_epoch})\n"
               f"Args: {vars(args)}\n")
    with open(os.path.join(args.save_dir, 'training_summary.txt'), 'w') as f:
        f.write(summary)
    print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SSMPhysics (physics-informed selective SSM Kalman filter)')

    # Data
    parser.add_argument('--dataset_root', type=str, default='./data/radiate/train', help='RADIATE train split')
    parser.add_argument('--min_seq_len', type=int, default=10, help='Minimum track length')
    parser.add_argument('--max_seq_len', type=int, default=10, help='Tracks are split into windows of this length')
    parser.add_argument('--obs_dim', type=int, default=5, help='Observation dimension [cx, cy, w, h, angle]')
    parser.add_argument('--state_dim', type=int, default=7, help='State dimension [cx, cy, vx, vy, w, h, angle]')

    # Model
    parser.add_argument('--d_model', type=int, default=256, help='Feature dimension')
    parser.add_argument('--d_state', type=int, default=32, help='SSM state dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of selective SSM layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--ssm_type', type=str, default='mamba1_cuda', choices=['mamba1_cuda', 'mamba1'],
                        help='mamba1_cuda: mamba_ssm CUDA kernels; mamba1: pure-PyTorch fallback')

    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)

    # Misc
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/ssm_physics_10frames')

    train(parser.parse_args())
