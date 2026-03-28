# This file is originally from DepthCrafter/depthcrafter/utils.py at main · Tencent/DepthCrafter
# SPDX-License-Identifier: MIT License license
#
# This file may have been modified by ByteDance Ltd. and/or its affiliates on [date of modification]
# Original file is released under [ MIT License license], with the full license text available at [https://github.com/Tencent/DepthCrafter?tab=License-1-ov-file].
import re
from pathlib import Path
import cv2

import numpy as np
import matplotlib.cm as cm
import imageio
try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except:
    import cv2
    DECORD_AVAILABLE = False

def ensure_even(value):
    return value if value % 2 == 0 else value + 1


def _natural_sort_key(path):
    name = path.name if isinstance(path, Path) else path
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def read_video_frames(video_path, process_length, target_fps=-1, max_res=-1, frame_seq=False):
    if frame_seq:
        import cv2

        folder = Path(video_path)
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
        paths = sorted(
            [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts],
            key=_natural_sort_key,
        )
        if not paths:
            raise FileNotFoundError(f"No image files found in frame folder: {video_path}")

        indices = list(range(len(paths)))
        if process_length != -1 and process_length < len(indices):
            indices = indices[:process_length]

        orig_hw_list = []
        frames_list = []
        target_height = target_width = None

        for j, i in enumerate(indices):
            raw = cv2.imread(str(paths[i]))
            if raw is None:
                raise ValueError(f"Failed to read image: {paths[i]}")
            frame = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            oh, ow = int(frame.shape[0]), int(frame.shape[1])
            orig_hw_list.append((oh, ow))

            if j == 0:
                original_height, original_width = oh, ow
                height, width = original_height, original_width
                if max_res > 0 and max(height, width) > max_res:
                    scale = max_res / max(original_height, original_width)
                    height = ensure_even(round(original_height * scale))
                    width = ensure_even(round(original_width * scale))
                target_height, target_width = height, width

            if frame.shape[0] != target_height or frame.shape[1] != target_width:
                frame = cv2.resize(frame, (target_width, target_height))
            frames_list.append(frame)

        frames = np.stack(frames_list, axis=0)
        frame_orig_hw = np.array(orig_hw_list, dtype=np.int32)
        fps = 30
        return frames, fps, frame_orig_hw

    if DECORD_AVAILABLE:
        vid = VideoReader(video_path, ctx=cpu(0))
        original_height, original_width = vid.get_batch([0]).shape[1:3]
        height = original_height
        width = original_width
        if max_res > 0 and max(height, width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = ensure_even(round(original_height * scale))
            width = ensure_even(round(original_width * scale))

        vid = VideoReader(video_path, ctx=cpu(0), width=width, height=height)

        fps = vid.get_avg_fps() if target_fps == -1 else target_fps
        stride = round(vid.get_avg_fps() / fps)
        stride = max(stride, 1)
        frames_idx = list(range(0, len(vid), stride))
        if process_length != -1 and process_length < len(frames_idx):
            frames_idx = frames_idx[:process_length]
        frames = vid.get_batch(frames_idx).asnumpy()
    else:
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        if max_res > 0 and max(original_height, original_width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = round(original_height * scale)
            width = round(original_width * scale)

        fps = original_fps if target_fps < 0 else target_fps

        stride = max(round(original_fps / fps), 1)

        frames = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (process_length > 0 and frame_count >= process_length):
                break
            if frame_count % stride == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                if max_res > 0 and max(original_height, original_width) > max_res:
                    frame = cv2.resize(frame, (width, height))  # Resize frame
                frames.append(frame)
            frame_count += 1
        cap.release()
        frames = np.stack(frames, axis=0)

    return frames, fps, None


def save_video(frames, output_video_path, fps=10, is_depths=False, grayscale=False):
    """
    Save a list/array of frames as a video, handling depth visualization
    and ensuring frame dimensions are compatible with H.264 (even numbers).

    Parameters:
        frames (np.ndarray): Array of frames, shape (N, H, W) or (N, H, W, C)
        output_video_path (str): Path to save the video
        fps (int): Frames per second
        is_depths (bool): If True, frames are depth maps and will be color-mapped
        grayscale (bool): If True and is_depths=True, use grayscale instead of colormap
    """
    
    def make_even(x):
        return x if x % 2 == 0 else x - 1

    # Determine target size from first frame
    first_frame = frames[0]
    h, w = first_frame.shape[:2]
    h, w = make_even(h), make_even(w)  # ensure even dimensions

    writer = imageio.get_writer(
        output_video_path,
        fps=fps,
        macro_block_size=1,
        codec='libx264',
        format='FFMPEG',
        ffmpeg_params=['-crf', '18']
    )

    if is_depths:
        colormap = np.array(cm.get_cmap("inferno").colors)
        d_min, d_max = frames.min(), frames.max()

    for i in range(frames.shape[0]):
        frame = frames[i]

        # Resize frame to target size
        frame = cv2.resize(frame, (w, h))

        if is_depths:
            depth_norm = ((frame - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            if grayscale:
                frame_out = depth_norm
            else:
                frame_out = (colormap[depth_norm] * 255).astype(np.uint8)
        else:
            frame_out = frame

        # Safety crop to ensure even shape
        frame_out = frame_out[:h, :w]

        writer.append_data(frame_out)

    writer.close()