"""Utility functions: KalmanNet helper and MSE metrics."""

import torch
import numpy as np


def expand_dim(x, seq_len_input, batch_size, device):
    """Expand dimension for GRU/LSTM input."""
    expanded = torch.empty(seq_len_input, batch_size, x.shape[-1]).to(device)
    expanded[0, :, :] = x
    return expanded


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_mse(pred, target, dim=None):
    """
    Compute MSE (Mean Squared Error).
    
    Args:
        pred: Predicted values [batch, seq_len, dim] or [seq_len, dim]
        target: Target values [batch, seq_len, dim] or [seq_len, dim]
        dim: Dimension to compute MSE over (None = all dimensions)
    
    Returns:
        MSE value (inf if pred contains inf/NaN)
    """
    # Check for inf/NaN
    if torch.isnan(pred).any() or torch.isinf(pred).any():
        return float('inf')
    
    if dim is not None:
        mse = torch.mean((pred - target) ** 2, dim=dim)
        # Check if mse contains inf/NaN
        if torch.isnan(mse).any() or torch.isinf(mse).any():
            return float('inf')
        return torch.mean(mse).item()
    else:
        mse = torch.mean((pred - target) ** 2)
        # Check if mse is inf/NaN
        if torch.isnan(mse) or torch.isinf(mse):
            return float('inf')
        return mse.item()


def mse_to_db(mse):
    """
    Convert MSE to dB scale.
    
    Args:
        mse: Mean Squared Error value
    
    Returns:
        MSE in dB: 10 * log10(MSE)
        Returns -inf if MSE is 0 or negative
    """
    if mse <= 0 or np.isinf(mse) or np.isnan(mse):
        return float('-inf')
    return 10.0 * np.log10(mse)


def compute_pos_mse(pred_state, target_state):
    """
    Compute Position MSE (cx, cy).
    
    Args:
        pred_state: [batch, seq_len, state_dim] or [seq_len, state_dim]
        target_state: [batch, seq_len, state_dim] or [seq_len, state_dim]
    
    Returns:
        Position MSE
    """
    pred_pos = pred_state[..., :2]  # [cx, cy]
    target_pos = target_state[..., :2]
    return compute_mse(pred_pos, target_pos)


def compute_vel_mse(pred_state, target_state):
    """
    Compute Velocity MSE (vx, vy).
    
    Args:
        pred_state: [batch, seq_len, state_dim] or [seq_len, state_dim]
        target_state: [batch, seq_len, state_dim] or [seq_len, state_dim]
    
    Returns:
        Velocity MSE
    """
    if pred_state.shape[-1] >= 4 and target_state.shape[-1] >= 4:
        pred_vel = pred_state[..., 2:4]  # [vx, vy]
        target_vel = target_state[..., 2:4]
        return compute_mse(pred_vel, target_vel)
    else:
        return 0.0  # No velocity in state


def compute_nstep_mse(pred_state, target_state, n=5):
    """
    Compute N-step ahead MSE (average MSE over last N steps).
    
    Args:
        pred_state: [batch, seq_len, state_dim] or [seq_len, state_dim]
        target_state: [batch, seq_len, state_dim] or [seq_len, state_dim]
        n: Number of steps to average over
    
    Returns:
        N-step MSE
    """
    if len(pred_state.shape) == 3:
        # Batch mode
        seq_len = pred_state.shape[1]
        if seq_len < n:
            return compute_mse(pred_state, target_state)
        pred_nstep = pred_state[:, -n:, :]
        target_nstep = target_state[:, -n:, :]
        return compute_mse(pred_nstep, target_nstep)
    else:
        # Single sequence
        seq_len = pred_state.shape[0]
        if seq_len < n:
            return compute_mse(pred_state, target_state)
        pred_nstep = pred_state[-n:, :]
        target_nstep = target_state[-n:, :]
        return compute_mse(pred_nstep, target_nstep)
