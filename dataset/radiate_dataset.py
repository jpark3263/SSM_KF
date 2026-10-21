"""Dataset loader for RADIATE dataset."""

import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np


class RadiateDataset(Dataset):
    """
    Dataset for RADIATE tracking.
    
    Converts bounding boxes to state/observation format:
    - Observation: [cx, cy, w, h, angle]
    - State: [cx, cy, vx, vy, w, h, angle]
    """
    
    def __init__(self, dataset_root, seq_names=None, min_seq_len=10, max_seq_len=100, 
                 obs_dim=5, state_dim=7, dt=1.0):
        """
        Initialize dataset.
        
        Args:
            dataset_root: Root directory of RADIATE dataset
            seq_names: List of sequence names to use (None = all)
            min_seq_len: Minimum sequence length
            max_seq_len: Maximum sequence length
            obs_dim: Observation dimension (4 or 5)
            state_dim: State dimension (6 or 7)
            dt: Time step (for velocity calculation)
        """
        self.dataset_root = dataset_root
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.dt = dt
        
        # Find all sequences
        if seq_names is None:
            seq_names = [d for d in os.listdir(dataset_root) 
                        if os.path.isdir(os.path.join(dataset_root, d)) 
                        and os.path.exists(os.path.join(dataset_root, d, 'annotations', 'annotations.json'))]
        
        self.sequences = []
        for seq_name in seq_names:
            seq_path = os.path.join(dataset_root, seq_name)
            ann_path = os.path.join(seq_path, 'annotations', 'annotations.json')
            if os.path.exists(ann_path):
                self.sequences.append(seq_path)
        
        # Load all tracks
        self.tracks = []
        for seq_path in self.sequences:
            ann_path = os.path.join(seq_path, 'annotations', 'annotations.json')
            with open(ann_path, 'r') as f:
                annotations = json.load(f)
            
            for obj in annotations:
                tracks = self._extract_track(obj)
                if tracks is not None:
                    for track in tracks:
                        if len(track) >= min_seq_len:
                            # Split long tracks into segments
                            for i in range(0, len(track), max_seq_len):
                                track_segment = track[i:i+max_seq_len]
                                if len(track_segment) >= min_seq_len:
                                    self.tracks.append(track_segment)
    
    def _extract_track(self, obj, max_gap=5):
        """
        Extract track from object annotation.
        
        Args:
            obj: Object annotation dict with 'bboxes' list
            max_gap: Maximum number of consecutive missing frames before splitting track
            
        Returns:
            List of tracks (each track is a list of [cx, cy, w, h, angle])
        """
        bboxes = obj['bboxes']
        tracks = []
        current_track = []
        gap_count = 0
        
        # Process each frame (ann_id = frame_id - 1, so frame_idx = ann_id)
        for ann_id, bbox in enumerate(bboxes):
            # Check if bbox exists and is valid (following radiate_plot.py pattern)
            if ann_id < len(bboxes) and bboxes[ann_id] and isinstance(bboxes[ann_id], dict):
                bbox_data = bboxes[ann_id]
                if 'position' in bbox_data:
                    pos = bbox_data['position']  # [x, y, w, h]
                    rot = bbox_data.get('rotation', 0.0)
                    
                    # Convert to center coordinates (following radiate_plot.py)
                    cx = pos[0] + pos[2] / 2.0
                    cy = pos[1] + pos[3] / 2.0
                    w = pos[2]
                    h = pos[3]
                    angle = rot
                    
                    current_track.append([cx, cy, w, h, angle])
                    gap_count = 0
                else:
                    # Missing position data
                    gap_count += 1
                    if gap_count > max_gap and len(current_track) > 0:
                        # Split track if gap is too large
                        if len(current_track) >= self.min_seq_len:
                            tracks.append(current_track)
                        current_track = []
                        gap_count = 0
            else:
                # Empty bbox (following radiate_plot.py: obj['bboxes'][ann_id] is empty list or None)
                gap_count += 1
                if gap_count > max_gap and len(current_track) > 0:
                    # Split track if gap is too large
                    if len(current_track) >= self.min_seq_len:
                        tracks.append(current_track)
                    current_track = []
                    gap_count = 0
        
        # Add final track
        if len(current_track) >= self.min_seq_len:
            tracks.append(current_track)
        
        return tracks if len(tracks) > 0 else None
    
    def _compute_velocity(self, track):
        """Compute velocity from position differences."""
        if len(track) < 2:
            return track
        
        state_track = []
        prev_pos = None
        
        for i, obs in enumerate(track):
            cx, cy, w, h, angle = obs
            
            if prev_pos is None:
                vx, vy = 0.0, 0.0
            else:
                vx = (cx - prev_pos[0]) / self.dt
                vy = (cy - prev_pos[1]) / self.dt
            
            # State: [cx, cy, vx, vy, w, h, angle]
            state = [cx, cy, vx, vy, w, h, angle]
            state_track.append(state)
            
            prev_pos = [cx, cy]
        
        return state_track
    
    def __len__(self):
        return len(self.tracks)
    
    def __getitem__(self, idx):
        track = self.tracks[idx]
        
        # Convert to numpy arrays
        obs_track = np.array(track, dtype=np.float32)  # [seq_len, 5]
        state_track = np.array(self._compute_velocity(track), dtype=np.float32)  # [seq_len, 7]
        
        # Truncate/pad to max_seq_len
        seq_len = len(obs_track)
        if seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - seq_len
            obs_track = np.pad(obs_track, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
            state_track = np.pad(state_track, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
        else:
            obs_track = obs_track[:self.max_seq_len]
            state_track = state_track[:self.max_seq_len]
            seq_len = self.max_seq_len
        
        # Select dimensions
        if self.obs_dim == 4:
            obs_track = obs_track[:, :4]  # [cx, cy, w, h]
        else:
            obs_track = obs_track[:, :5]  # [cx, cy, w, h, angle]
        
        if self.state_dim == 6:
            state_track = state_track[:, :6]  # [cx, cy, vx, vy, w, h]
        else:
            state_track = state_track[:, :7]  # [cx, cy, vx, vy, w, h, angle]
        
        return {
            'observation': torch.FloatTensor(obs_track),  # [seq_len, obs_dim]
            'state': torch.FloatTensor(state_track),  # [seq_len, state_dim]
            'seq_len': seq_len
        }

