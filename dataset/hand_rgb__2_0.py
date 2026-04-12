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


class HandRGBDataset(Dataset):
    def __init__(
        self,
        image_size: Tuple[int, int] = (512, 512),
        extra_anno_keys: Sequence[Literal["extrinsics"]] = [],
        sequence_length: int = 1,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.extra_anno_keys = set(extra_anno_keys) if extra_anno_keys is not None else None
        self.sequence_length = sequence_length

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
        self.build_sequence_windows()

        if len(self.sequence_windows) == 0:
            raise ValueError("No valid sequence windows found.")
        
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
        
        if sequence_cache_file is not None and sequence_cache_file.exists():
            print(f"[INFO] Loading sequences from cache {sequence_cache_file}.")

            with open(sequence_cache_file, "rb") as f:
                sequences = pickle.load(f)
        else:
            sequences = self.collect_sequences()

            if sequence_cache_file is not None:
                sequence_cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(sequence_cache_file, "wb") as f:
                    pickle.dump(sequences, f, protocol=pickle.HIGHEST_PROTOCOL)

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

    def build_sequence_windows(self):
        self.sequence_windows = []

        if self.sequence_length <= 1:
            for frame_idx in range(len(self.frames)):
                self.sequence_windows.append({
                    "mode": "single",
                    "frame_idx": frame_idx,
                })
            return

        for seq in self.seqs:
            seq_length = seq["length"]
            if seq_length < self.sequence_length:
                continue

            self.sequence_windows.append({
                "mode": "temporal",
                "seq_id": seq["seq_id"],
                "seq_start": seq["start"],
                "max_offset": seq_length - self.sequence_length,
            })

        print(
            f"[INFO] Built {len(self.sequence_windows)} windows with sequence_length={self.sequence_length}."
        )
    
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
        return len(self.sequence_windows)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        window = self.sequence_windows[idx]

        if window["mode"] == "single":
            frame = self.frames[window["frame_idx"]]
            return self._build_single_sample(frame=frame, index=idx)

        max_offset = window["max_offset"]
        if getattr(self, "split", None) == "train":
            offset = self.random.randint(0, max_offset)
        else:
            offset = 0

        start = window["seq_start"] + offset
        clip_frames = self.frames[start:start + self.sequence_length]

        annos = [self.read_annos(frame) for frame in clip_frames]
        annos = self.convert_sequence_annos_to_canonical(clip_frames, annos)
        image_dicts = [self.read_image(frame, anno) for frame, anno in zip(clip_frames, annos)]

        hand_side = torch.tensor([anno["hand_side"] for anno in annos], dtype=torch.long)
        hand_side_mask = torch.stack([anno["hand_side_mask"] for anno in annos], dim=0)

        sample = {
            "hand_side": hand_side,
            "hand_side_mask": hand_side_mask,
            "object": annos[0]["object"],
            "image": torch.stack([img["image"] for img in image_dicts], dim=0),
            "left_hand_image": torch.stack([img["left_hand_image"] for img in image_dicts], dim=0),
            "left_crop_trans": torch.stack([img["left_crop_trans"] for img in image_dicts], dim=0),
            "right_hand_image": torch.stack([img["right_hand_image"] for img in image_dicts], dim=0),
            "right_crop_trans": torch.stack([img["right_crop_trans"] for img in image_dicts], dim=0),
            "betas": torch.stack([anno["betas"] for anno in annos], dim=0),
            "pose": torch.stack([anno["pose"] for anno in annos], dim=0),
            "transl": torch.stack([anno["transl"] for anno in annos], dim=0),
            "joints3d": torch.stack([anno["joints3d"] for anno in annos], dim=0),
            "intrinsics": torch.stack([anno["intrinsics"] for anno in annos], dim=0),
            "intrinsics_resize": torch.stack([anno["intrinsics_resize"] for anno in annos], dim=0),
            "extrinsics": torch.stack([anno["extrinsics"] for anno in annos], dim=0),
            "meta": {
                "index": idx,
                "seq_id": clip_frames[0].get("seq_id", None),
                "t_start": clip_frames[0].get("t", None),
                "frame_ids": [frame.get("frame_id", None) for frame in clip_frames],
                "t": [frame.get("t", None) for frame in clip_frames],
            },
        }

        return sample

    def _build_single_sample(self, frame: Dict[str, Any], index: int) -> Dict[str, Any]:
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

    def convert_sequence_annos_to_canonical(
        self,
        frames: Sequence[Dict[str, Any]],
        annos: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return list(annos)
    
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
