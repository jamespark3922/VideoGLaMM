from pathlib import Path
from glob import glob
import os
import random
from typing import List, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from datasets import load_from_disk



def subsample_images(images, t):
    if isinstance(images, list):
        num_images = len(images)
        if t < num_images:
            indices = np.linspace(0, num_images - 1, num=t, dtype=int)
            return [images[i] for i in indices]
        else:
            return images
    elif isinstance(images, np.ndarray):
        T = images.shape[0]
        if t < T:
            indices = np.linspace(0, T - 1, num=t, dtype=int)
            return images[indices]
        else:
            return images
    else:
        raise ValueError("Input images must be either a list of PIL images or a numpy array.")

def get_imgs_from_video(data_i):
    
    # video_annotations = data_i['annotations']
    
    # assert len(video_annotations)==data_i['length'], f"len(video_annotations): {len(video_annotations)}     data_i['length']:{data_i['length']}"
    vid_len = data_i['length']
    
    # h,w = data_i['annotations'][0][0]['segmentation']['size']
    imgs = []
    for frame_idx in range(vid_len):
        img_path = data_i['file_names'][frame_idx]
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        imgs.append(img)
    
    return imgs

class ProlificDataset(Dataset):
    """
    Prolific dataset, but:
      - Expressions & video ids come from a HuggingFace dataset.
      - Frames are loaded from a directory tree.

    Returns: np_images, target
        np_images: [T, H, W, 3], uint8
        target: {
            'frames_idx': Tensor[T],
            'caption': str,
            'orig_size': Tensor[2] (H, W),
            'size': Tensor[2] (H, W),
            'pil_images': list of PIL.Image
        }
    """

    def __init__(self, video_frames_dir, hf_meta_path, transform=None):
        """
        HF-backed Refer-YouTube-VOS-style dataset.

        Args:
            hf_meta_path (str): Path to HuggingFace dataset on disk (load_from_disk),
                                e.g. the same path you pass as --meta_exp_path.
            video_frames_dir (str): Root dir with per-video frame folders,
                                    e.g. .../Ref-YT-VOS/valid/JPEGImages
                                    ==> video_frames_dir/<video_id>/*.jpg
            transform (callable, optional): Transform applied to each PIL image
                                            (e.g. ToTensor+normalize).
        """
        self.hf_meta_path = hf_meta_path
        self.video_frames_dir = video_frames_dir
        self.transform = transform

        # Load HF dataset (must have at least 'id' and 'exp')
        self.meta_ds = load_from_disk(self.hf_meta_path)

        # Build flat metas list similar to your previous class
        self.metas = []
        for i in range(len(self.meta_ds)):
            row = self.meta_ds[i]
            qid = row["id"]          # e.g. "videoId_00001"
            exp_text = row["exp"]

            # Try to get video id from a dedicated column if it exists
            if "video" in self.meta_ds.column_names:
                video_id = row["video"]
                # exp_id can still be derived from id if you want
                exp_id = qid.split("_")[-1]
            else:
                # Fallback: parse from id = "<video>_<expression_id>"
                # This matches your inference script: vid_id, exp_id = id.rsplit('_', 1)
                video_id, exp_id = qid.rsplit("_", 1)

            # Collect all frame names for this video from disk
            video_dir = Path(self.video_frames_dir) / video_id
            if not video_dir.exists():
                raise FileNotFoundError(f"Video directory not found: {video_dir}")

            frame_files = sorted(
                glob(os.path.join(str(video_dir), "*.jpg"))
                + glob(os.path.join(str(video_dir), "*.png"))
            )
            if len(frame_files) == 0:
                raise RuntimeError(f"No frames found for video {video_id} in {video_dir}")

            frames = [Path(f).stem for f in frame_files]

            self.metas.append(
                {
                    "video": video_id,
                    "id": qid,
                    "expression_id": exp_id,
                    "expression": exp_text,
                    "frames": frames,
                }
            )

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        """
        Returns:
            np_images: [T, H, W, 3] uint8
            target: {
                'caption': str,
                'pil_images': list[Tensor or PIL],  # see below
                'video_path': (video_id, expression_id),
                'frame_ids': list[str],
            }
        """
        meta = self.metas[idx]
        video = meta["video"]
        expression_id = meta["expression_id"]
        expression_text = meta["expression"]
        frames = meta["frames"]

        # Build frame paths
        img_paths = [
            os.path.join(self.video_frames_dir, video, frame + ".jpg") for frame in frames
        ]
        # Fallback for .png if .jpg doesn’t exist
        img_paths = [
            p if os.path.exists(p) else p.replace(".jpg", ".png") for p in img_paths
        ]

        # Load PIL images
        pil_imgs = [Image.open(p).convert("RGB") for p in img_paths]

        # Apply transform if provided
        if self.transform is not None:
            imgs = [self.transform(img) for img in pil_imgs]  # list of tensors [3, H, W]
            imgs = torch.stack(imgs, dim=0)  # [T, 3, H, W]
            # For target['pil_images'] you may want the transformed tensors:
            pil_images_for_target = imgs
        else:
            # No transform: keep PIL images in target
            pil_images_for_target = pil_imgs

        # For np_images, return raw uint8 RGB frames in THWC
        np_images = np.stack([np.array(img) for img in pil_imgs], axis=0)  # [T, H, W, 3]

        target = {
            "caption": expression_text,
            "pil_images": pil_images_for_target,
            "video_path": (video, expression_id),
            "frame_ids": frames,
        }

        return np_images, target
