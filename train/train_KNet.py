"""Train the KalmanNet baseline on RADIATE tracks.

Paper setting:
    python train/train_KNet.py --dataset_root data/radiate/train --use_cuda
"""

import os
import sys
import argparse
import csv
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.kalmannet import KalmanNetNN
from model.system_model import TrackingSystemModel
from dataset.radiate_dataset import RadiateDataset
from utils.utils import count_parameters
from run import Args


def run_sequences(model, obs, state, seq_len, criterion):
    """Filter each sequence of the batch frame by frame; return the mean per-sequence MSE."""
    loss_batch = 0.0
    for b in range(obs.shape[0]):
        seq_len_b = seq_len[b].item()
        obs_seq = obs[b, :seq_len_b, :]  # [seq_len, obs_dim]
        state_seq = state[b, :seq_len_b, :]  # [seq_len, state_dim]

        model.batch_size = 1
        model.InitSequence(state_seq[0:1, :].transpose(0, 1).unsqueeze(0), seq_len_b)  # GT initial state
        model.init_hidden_KNet()

        pred_states = []
        for t in range(seq_len_b):
            y_t = obs_seq[t:t+1, :].transpose(0, 1).unsqueeze(0)  # [1, obs_dim, 1]
            pred_states.append(model(y_t).squeeze(0).squeeze(-1))
        loss_batch += criterion(torch.stack(pred_states, dim=0), state_seq)
    return loss_batch / obs.shape[0]


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print(f'Using device: {device}')
    torch.manual_seed(args.seed)

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
    model = KalmanNetNN()
    model.NNBuild(sys_model, Args(use_cuda=device.type == 'cuda', n_batch=args.batch_size,
                                  in_mult_KNet=args.in_mult_KNet, out_mult_KNet=args.out_mult_KNet))
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    num_params = count_parameters(model)
    print(f'Parameters: {num_params:,}')
    print(f'Train / val sequences: {len(train_dataset)} / {len(val_dataset)}')

    best_val_loss = float('inf')
    best_epoch = 0
    os.makedirs(args.save_dir, exist_ok=True)
    log_csv_path = os.path.join(args.save_dir, 'training_log.csv')
    with open(log_csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'learning_rate'])

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]'):
            optimizer.zero_grad()
            loss = run_sequences(model, batch['observation'].to(device), batch['state'].to(device),
                                 batch['seq_len'], criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                val_loss += run_sequences(model, batch['observation'].to(device), batch['state'].to(device),
                                          batch['seq_len'], criterion).item()
        val_loss /= len(val_loader)
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch+1}: train {train_loss:.4f}, val {val_loss:.4f}, lr {current_lr:.6f}')
        with open(log_csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, train_loss, val_loss, current_lr])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_args': vars(args),
            }, os.path.join(args.save_dir, 'best_model.pth'))

    summary = (f"Model: KalmanNetNN ({num_params:,} parameters)\n"
               f"Finished: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
               f"Train / val sequences: {len(train_dataset)} / {len(val_dataset)}\n"
               f"Best val loss: {best_val_loss:.6f} (epoch {best_epoch})\n"
               f"Args: {vars(args)}\n")
    with open(os.path.join(args.save_dir, 'training_summary.txt'), 'w') as f:
        f.write(summary)
    print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train KalmanNet baseline')

    # Data
    parser.add_argument('--dataset_root', type=str, default='./data/radiate/train', help='RADIATE train split')
    parser.add_argument('--min_seq_len', type=int, default=10, help='Minimum track length')
    parser.add_argument('--max_seq_len', type=int, default=100, help='Tracks are split into windows of this length')
    parser.add_argument('--obs_dim', type=int, default=5)
    parser.add_argument('--state_dim', type=int, default=7)

    # Model (KalmanNet_TSP defaults)
    parser.add_argument('--in_mult_KNet', type=int, default=5)
    parser.add_argument('--out_mult_KNet', type=int, default=40)

    # Training (released checkpoint: lr 1e-3, 25 epochs, batch 8)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_step', type=int, default=20, help='StepLR step size')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='StepLR gamma')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)

    # Misc
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/knet')

    train(parser.parse_args())
