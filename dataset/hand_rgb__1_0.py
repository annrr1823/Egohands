from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Literal

import os
import random

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md"],
    pythonpath=True,
    dotenv=True,
)

import numpy as np
from pathlib import Path
import pickle
import torch
from torch.utils.data import Dataset

from model.smpl.mano import MANOLayer, MANO_PATH
from dataset.utils.utils import _pad_clip, _temporal_stack


class HandRGBDataset(Dataset):
    def __init__(
        self,
        image_size: Tuple[int, int] = (512, 512),
        extra_anno_keys: Sequence[Literal["extrinsics"]] = [],
        clip_len: int = 1,
        random_clip_for_train: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.extra_anno_keys = set(extra_anno_keys) if extra_anno_keys is not None else None
        self.clip_len = int(clip_len)
        self.random_clip_for_train = bool(random_clip_for_train)

        if self.clip_len <= 0:
            raise ValueError(f"clip_len must be positive, got {self.clip_len}.")

        self.image_size = image_size
        self.image_mean = np.array([0.485, 0.456, 0.406])
        self.image_std = np.array([0.229, 0.224, 0.225])

        if self.raw_image_size is not None:
            raw_width, raw_height = self.raw_image_size
            resize_width, resize_height = self.image_size
            self.resize_scale = [resize_width / raw_width, resize_height / raw_height]
        else:
            self.resize_scale = [1.0, 1.0]

        self.random = random.Random(seed)
        
        self.mano_lhand = MANOLayer(model_path=MANO_PATH, is_rhand=False)
        self.mano_rhand = MANOLayer(model_path=MANO_PATH, is_rhand=True)

        self.load_metadata()

        if len(self.frames) == 0:
            raise ValueError("Frames is empty.")
        
    def load_metadata(self):
        """
        self.frames = [
            {
            "seq_id": str,
            "t": int,
            "frame_id": str,
            },
            ...
        ]

        self.seqs = [
            {
            "seq_id": str,
            "start": int,    # index into frames
            "length": int,   # number of frames
            },
            ...
        ]
        self.seqs_map = {
            "seq_id": seq_idx_in_seqs
        }
        """
        sequence_cache_file = getattr(self, "sequence_cache_file", None)
        sequence_cache_file = Path(sequence_cache_file) if sequence_cache_file else None
        
        sequences = None
        if sequence_cache_file is not None and sequence_cache_file.exists():
            print(f"[INFO] Loading sequences from cache {sequence_cache_file}.")
            try:
                with open(sequence_cache_file, "rb") as f:
                    sequences = pickle.load(f)
            except (EOFError, pickle.UnpicklingError) as exc:
                print(f"[WARN] Failed to load cache {sequence_cache_file}: {exc}. Rebuilding cache.")

        if sequences is None:
            sequences = self.collect_sequences()

            if sequence_cache_file is not None:
                sequence_cache_file.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = sequence_cache_file.with_suffix(sequence_cache_file.suffix + ".tmp")
                with open(tmp_path, "wb") as f:
                    pickle.dump(sequences, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp_path, sequence_cache_file)
                print(f"[INFO] Saved sequences cache to {sequence_cache_file}.")

        self.frames = []
        self.seqs = []
        self.seqs_map = {}

        for seq in sequences:
            seq_id = str(seq.get("seq_id", len(self.seqs)))
            frame_ids = seq["frame_ids"]

            start = len(self.frames)
            length = len(frame_ids)
            if length <= 0:
                continue

            for i, frame_id in enumerate(frame_ids):
                self.frames.append({
                    "seq_id": seq_id,
                    "t": i,
                    "frame_id": frame_id,
                })

            self.seqs.append({
                "seq_id": seq_id,
                "start": start,
                "length": length,
            })
            self.seqs_map[seq_id] = len(self.seqs) - 1
        
        print(f"[INFO] Collected {len(self.seqs)} sequences, {len(self.frames)} frames.")
    
    def collect_sequences(self) -> List[dict]:
        """
        Return a list of sequences, each like:
            {
                "seq_id": "S0001",
                "frame_ids": [
                    "/path/to/image_0001.jpg",
                    "/path/to/image_0002.jpg",
                    ...
                ]
            }
        """
        raise NotImplementedError
    
    def __len__(self) -> int:
        return len(self.frames)

    def _sample_clip_frames(self, idx: int) -> List[Dict[str, Any]]:
        frame = self.frames[idx]
        if self.clip_len == 1:
            return [frame]

        seq_id = frame["seq_id"]
        seq_idx = self.seqs_map[seq_id]
        seq_info = self.seqs[seq_idx]

        seq_start = int(seq_info["start"])
        seq_length = int(seq_info["length"])
        rel_t = int(frame["t"])

        max_start = max(seq_length - self.clip_len, 0)
        if getattr(self, "split", "train") == "train" and self.random_clip_for_train and max_start > 0:
            start_rel = self.random.randint(0, max_start)
        else:
            start_rel = min(max(rel_t - self.clip_len // 2, 0), max_start)

        clip = []
        clip_end = min(start_rel + self.clip_len, seq_length)
        for t in range(start_rel, clip_end):
            clip.append(self.frames[seq_start + t])

        clip = _pad_clip(clip, target_len=self.clip_len, pad_mode="repeat")
        return clip

    def _load_single_frame(self, frame: Dict[str, Any], index: int) -> Dict[str, Any]:
        anno = self.read_annos(frame)
        image_dict = self.read_image(frame, anno)

        frame_meta = {
            "frame_id": frame.get("frame_id", None),
            "seq_id": frame.get("seq_id", None),
            "t": frame.get("t", None),
        }

        sample = {
            "hand_side": anno["hand_side"],
            "hand_side_mask": anno["hand_side_mask"],
            "mano_side_mask": anno.get("mano_side_mask", anno["hand_side_mask"]),
            "object": anno["object"],
            "image": image_dict["image"],
            "left_hand_image": image_dict["left_hand_image"],
            "left_crop_trans": image_dict["left_crop_trans"],
            "right_hand_image": image_dict["right_hand_image"],
            "right_crop_trans": image_dict["right_crop_trans"],
            "betas": anno["betas"],
            "pose": anno["pose"],
            "transl": anno["transl"],
            "joints3d": anno["joints3d"],
            "intrinsics": anno["intrinsics"],
            "intrinsics_resize": anno["intrinsics_resize"],
            "extrinsics": anno["extrinsics"],
            "meta": {
                "index": index,
                **frame_meta,
            },
        }

        return sample
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        clip_frames = self._sample_clip_frames(idx)
        clip_samples = [self._load_single_frame(frame, idx) for frame in clip_frames]

        if self.clip_len == 1:
            return clip_samples[0]

        sample = {}
        keys = clip_samples[0].keys()
        for key in keys:
            if key == "meta":
                sample[key] = [item[key] for item in clip_samples]
            else:
                sample[key] = _temporal_stack([item[key] for item in clip_samples])

        return sample
    
    def convert_frame_id_to_image_path(self, frame_id: str):
        image_path = frame_id

        return image_path
    
    def read_annos(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def read_image(self, frame: Dict[str, Any], anno: Dict[str, Any]):
        raise NotImplementedError

    
    
from torch.utils.data._utils.collate import default_collate

def collate_fn_rgb(batch):
    not_stack_keys = {"meta"}
    keys = batch[0].keys()

    collation = {}
    for key in keys:
        values = [sample[key] for sample in batch]

        if key in not_stack_keys:
            collation[key] = values
            continue

        collation[key] = default_collate(values)
        
    return collation
