"""Fig. 5: qualitative comparison on RADIATE radar frames.

Runs the models on every annotated object track (>= 10 frames) of a test sequence and draws the
predicted rotated boxes on the Navtech_Cartesian radar images. Colours: GT green, KalmanNet red,
SSMPhysics (ours) yellow, KalmanFilter blue. Writes one PNG per frame and an MP4.
"""

import os
import sys
import json
import argparse

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import load_model, add_common_args, checkpoint_for, make_device

COLORS = {  # RGB
    'GT': (0, 255, 0),
    'KalmanFilter': (0, 0, 255),
    'KalmanNet': (255, 0, 0),
    'SSMPhysics': (255, 255, 0),
}


def state_to_bbox(state):
    """[cx, cy, vx, vy, w, h, angle] -> [x, y, w, h, angle] with (x, y) the top-left corner."""
    cx, cy, _, _, w, h, angle = state[:7]
    return [cx - w / 2.0, cy - h / 2.0, w, h, angle]


def draw_rotated_box(im, bbox, angle_deg, color, thickness=2, label=None):
    """Draw a box given as [x, y, w, h] (top-left corner) rotated by angle_deg (RADIATE convention)."""
    x, y, w, h = bbox[:4]
    theta = np.deg2rad(-angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]).T
    center = np.array([[x + w / 2.0], [y + h / 2.0]])
    corners = ((R @ corners) + center).T.astype(int)
    for i in range(4):
        cv2.line(im, tuple(int(v) for v in corners[i]), tuple(int(v) for v in corners[(i + 1) % 4]), color, thickness)
    if label is not None:
        cv2.putText(im, label, (int(corners[0][0]), int(corners[0][1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def create_video(output_dir, name, fps=10):
    frame_files = sorted(f for f in os.listdir(output_dir) if f.startswith('frame_') and f.endswith('.png'))
    if not frame_files:
        return
    height, width, _ = cv2.imread(os.path.join(output_dir, frame_files[0])).shape
    video_path = os.path.join(output_dir, f'{name}_visualization.mp4')
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for f in frame_files:
        frame = cv2.imread(os.path.join(output_dir, f))
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
    writer.release()
    print(f"Created video: {video_path}")


def extract_object_track(obj, min_seq_len=10):
    """Return ([cx, cy, w, h, angle] per annotated frame, annotation indices), or (None, None) if too short."""
    track, frame_indices = [], []
    for ann_id, bbox in enumerate(obj['bboxes']):
        if bbox and isinstance(bbox, dict) and 'position' in bbox:
            x, y, w, h = bbox['position']
            track.append([x + w / 2.0, y + h / 2.0, w, h, bbox.get('rotation', 0.0)])
            frame_indices.append(ann_id)
    if len(track) < min_seq_len:
        return None, None
    return track, frame_indices


def track_to_state(track, dt=1.0):
    """Ground-truth state with finite-difference velocity (first frame v = 0), as in RadiateDataset."""
    states, prev = [], None
    for cx, cy, w, h, angle in track:
        vx, vy = (0.0, 0.0) if prev is None else ((cx - prev[0]) / dt, (cy - prev[1]) / dt)
        states.append([cx, cy, vx, vy, w, h, angle])
        prev = (cx, cy)
    return states


@torch.no_grad()
def predict(model, model_name, obs_seq, state_seq):
    """Same inference protocol as run.evaluate_model."""
    device = obs_seq.device
    if model_name == 'SSMPhysics':
        prev_state = torch.cat([torch.zeros(1, state_seq.shape[1], device=device), state_seq[:-1]], dim=0)
        return model(obs_seq.unsqueeze(0), init_state=state_seq[0], prev_state=prev_state.unsqueeze(0)).squeeze(0)
    m1x_0 = state_seq[0:1, :].transpose(0, 1).unsqueeze(0)
    if model_name == 'KalmanNet':
        model.batch_size = 1
        model.InitSequence(m1x_0, len(obs_seq))
        model.init_hidden_KNet()
        return torch.stack([model(obs_seq[t:t+1].transpose(0, 1).unsqueeze(0)).squeeze(0).squeeze(-1)
                            for t in range(len(obs_seq))], dim=0)
    return model(obs_seq.unsqueeze(0), m1x_0=m1x_0)[0]


def visualize_sequence(sequence_path, models, device, output_dir, max_frames=None, fps=10):
    with open(os.path.join(sequence_path, 'annotations', 'annotations.json')) as f:
        annotations = json.load(f)
    radar_dir = os.path.join(sequence_path, 'Navtech_Cartesian')
    radar_files = sorted(f for f in os.listdir(radar_dir) if f.endswith('.png'))
    if max_frames:
        radar_files = radar_files[:max_frames]

    # Predictions per object track
    tracks, preds = {}, {}
    for obj_idx, obj in enumerate(annotations):
        track, frame_indices = extract_object_track(obj)
        if track is None:
            continue
        tracks[obj_idx] = frame_indices
        obs_seq = torch.tensor(track, dtype=torch.float32, device=device)
        state_seq = torch.tensor(track_to_state(track), dtype=torch.float32, device=device)
        preds[obj_idx] = {name: predict(m, name, obs_seq, state_seq).cpu().numpy() for name, m in models.items()}
    print(f"Found {len(tracks)} objects with tracks of >= 10 frames")

    os.makedirs(output_dir, exist_ok=True)
    for radar_file in radar_files:
        frame_id = int(os.path.splitext(radar_file)[0])
        ann_id = frame_id - 1  # annotation index (0-based)
        im = cv2.imread(os.path.join(radar_dir, radar_file))
        if im is None:
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        for obj in annotations:
            if ann_id < len(obj['bboxes']) and obj['bboxes'][ann_id]:
                gt = obj['bboxes'][ann_id]
                draw_rotated_box(im, gt['position'], gt.get('rotation', 0.0), COLORS['GT'], label='GT')

        for obj_idx, frame_indices in tracks.items():
            if ann_id not in frame_indices:
                continue
            t = frame_indices.index(ann_id)
            for name, pred_seq in preds[obj_idx].items():
                bbox = state_to_bbox(pred_seq[t])
                draw_rotated_box(im, bbox[:4], bbox[4], COLORS[name], label=name)

        plt.figure(figsize=(12, 8))
        plt.imshow(im)
        plt.title(f'Frame {frame_id} - {os.path.basename(sequence_path)}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'frame_{frame_id:06d}.png'), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Saved {len(radar_files)} frames to {output_dir}")
    create_video(output_dir, os.path.basename(sequence_path), fps=fps)


def main():
    parser = add_common_args(argparse.ArgumentParser(description='Fig. 5: draw predicted boxes on RADIATE radar frames'))
    parser.set_defaults(models=['KalmanNet', 'SSMPhysics'])
    parser.add_argument('--test_seq_names', type=str, nargs='+', default=['city_7_0'],
                        help='Test sequences to visualise (Fig. 5 uses city_7_0)')
    parser.add_argument('--output_dir', type=str, default='./results/fig5')
    parser.add_argument('--max_frames', type=int, default=None, help='Visualise only the first N frames')
    parser.add_argument('--fps', type=int, default=10, help='Frame rate of the output video')
    args = parser.parse_args()
    device = make_device(args)

    models = {name: load_model(name, checkpoint_for(name, args), args, device).eval() for name in args.models}

    for seq_name in args.test_seq_names:
        print(f"\nVisualizing sequence: {seq_name}")
        visualize_sequence(os.path.join(args.test_dataset_root, seq_name), models, device,
                           os.path.join(args.output_dir, seq_name), args.max_frames, args.fps)


if __name__ == '__main__':
    main()
