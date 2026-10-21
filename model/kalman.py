"""Basic Linear Kalman Filter for tracking bounding boxes in RADIATE dataset.
"""

import torch
import torch.nn as nn


class KalmanFilter(nn.Module):
    """
    Linear Kalman Filter for tracking.
    
    For RADIATE dataset tracking:
    - State: [cx, cy, vx, vy, w, h, angle] (7D) or [cx, cy, vx, vy, w, h] (6D)
    - Observation: [cx, cy, w, h, angle] (5D) or [cx, cy, w, h] (4D)
    """
    
    def __init__(self, F, H, Q, R, device='cpu'):
        """
        Initialize Kalman Filter.
        
        Args:
            F: State transition matrix [state_dim, state_dim]
            H: Observation matrix [obs_dim, state_dim]
            Q: Process noise covariance [state_dim, state_dim]
            R: Observation noise covariance [obs_dim, obs_dim]
            device: Device to run on
        """
        super().__init__()
        self.device = device
        
        self.F = F.to(device)  # State transition matrix
        self.H = H.to(device)  # Observation matrix
        self.Q = Q.to(device)  # Process noise
        self.R = R.to(device)  # Observation noise
        
        self.state_dim = F.shape[0]
        self.obs_dim = H.shape[0]
        
    def init_sequence(self, m1x_0, m2x_0):
        """
        Initialize tracking sequence.
        
        Args:
            m1x_0: Initial state mean [batch_size, state_dim, 1]
            m2x_0: Initial state covariance [batch_size, state_dim, state_dim]
        """
        self.m1x_posterior = m1x_0.to(self.device)
        self.m2x_posterior = m2x_0.to(self.device)
        
    def predict(self):
        """Predict step: compute prior state estimate."""
        # Predict the 1-st moment of x
        self.m1x_prior = torch.bmm(self.batched_F, self.m1x_posterior)
        
        # Predict the 2-nd moment of x
        self.m2x_prior = torch.bmm(self.batched_F, self.m2x_posterior)
        self.m2x_prior = torch.bmm(self.m2x_prior, self.batched_F_T) + self.batched_Q
        
        # Predict the 1-st moment of y
        self.m1y = torch.bmm(self.batched_H, self.m1x_prior)
        
        # Predict the 2-nd moment of y
        self.m2y = torch.bmm(self.batched_H, self.m2x_prior)
        self.m2y = torch.bmm(self.m2y, self.batched_H_T) + self.batched_R
        
    def compute_kalman_gain(self):
        """Compute Kalman Gain."""
        self.KG = torch.bmm(self.m2x_prior, self.batched_H_T)
        self.KG = torch.bmm(self.KG, torch.inverse(self.m2y))
        
    def innovation(self, y):
        """Compute innovation (measurement residual)."""
        self.dy = y - self.m1y
        
    def correct(self):
        """Correction step: compute posterior state estimate."""
        # Compute the 1-st posterior moment
        self.m1x_posterior = self.m1x_prior + torch.bmm(self.KG, self.dy)
        
        # Compute the 2-nd posterior moment
        self.m2x_posterior = torch.bmm(self.m2y, torch.transpose(self.KG, 1, 2))
        self.m2x_posterior = self.m2x_prior - torch.bmm(self.KG, self.m2x_posterior)
        
    def forward(self, y, m1x_0=None, m2x_0=None):
        """
        Forward pass through Kalman Filter.
        
        Args:
            y: Observations [batch_size, seq_len, obs_dim]
            m1x_0: Initial state mean [batch_size, state_dim, 1] (optional)
            m2x_0: Initial state covariance [batch_size, state_dim, state_dim] (optional)
            
        Returns:
            x: State estimates [batch_size, seq_len, state_dim]
        """
        batch_size, seq_len, _ = y.shape
        y = y.to(self.device)
        
        # Prepare batched matrices
        self.batched_F = self.F.view(1, self.state_dim, self.state_dim).expand(batch_size, -1, -1)
        self.batched_F_T = torch.transpose(self.batched_F, 1, 2)
        self.batched_H = self.H.view(1, self.obs_dim, self.state_dim).expand(batch_size, -1, -1)
        self.batched_H_T = torch.transpose(self.batched_H, 1, 2)
        self.batched_Q = self.Q.view(1, self.state_dim, self.state_dim).expand(batch_size, -1, -1)
        self.batched_R = self.R.view(1, self.obs_dim, self.obs_dim).expand(batch_size, -1, -1)
        
        # Initialize state
        if m1x_0 is None:
            m1x_0 = torch.zeros(batch_size, self.state_dim, 1, device=self.device)
        if m2x_0 is None:
            m2x_0 = torch.eye(self.state_dim, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
        
        self.init_sequence(m1x_0, m2x_0)
        
        # Allocate output array
        x = torch.zeros(batch_size, seq_len, self.state_dim, device=self.device)
        
        # Process sequence
        for t in range(seq_len):
            yt = y[:, t:t+1, :].transpose(1, 2)  # [batch_size, obs_dim, 1]
            
            # Kalman Filter steps
            self.predict()
            self.compute_kalman_gain()
            self.innovation(yt)
            self.correct()
            
            # Store result
            x[:, t, :] = self.m1x_posterior.squeeze(-1)
            
        return x
