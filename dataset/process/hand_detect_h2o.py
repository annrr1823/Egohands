from pathlib import Path

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md"],
    pythonpath=True,
    dotenv=True,
)

import numpy as np
from tqdm import tqdm

from dataset.h2o__1_0 import H2OEgoRGBDataset
from dataset.process.utils.track_hand import track_hand


if __name__ == "__main__":
    dataset = H2OEgoRGBDataset(
        dataset_folder="/data/public/dataset/hand/h2o",
        split="train",
        setup="default",
        image_size=(1280, 720),
    )

    frames = dataset.frames
    seqs = dataset.seqs

    for seq in tqdm(seqs):
        start = seq["start"]
        length = seq["length"]

        image_paths = []
        for frame in frames[start:start + length]:
            image_path = frame["frame_id"]
            image_paths.append(image_path)
        
        track_result = track_hand(image_paths, track_conf=0.6)

        parts = Path(image_paths[0]).parts
        hand_bbox_path = dataset.cache_dir / "hand_detection" / "h2o" / parts[-6] / parts[-5] / parts[-4] / "bbox.npy"
        hand_joints2d_path = dataset.cache_dir / "hand_detection" / "h2o" / parts[-6] / parts[-5] / parts[-4] / "joints2d.npy"
        hand_bbox_path.parent.mkdir(parents=True, exist_ok=True)
        
        bbox = np.ones((2, length, 5), dtype=np.float32) * -1
        joints2d = np.ones((2, length, 21, 3), dtype=np.float32) * -1

        for track_id, track_info in track_result.items():
            hand_side = track_info["hand_side"]
            if hand_side not in [0, 1]:
                continue

            for frame_index, box_with_conf in track_info["box_with_conf"].items():
                if 0 <= frame_index < length:
                    bbox[hand_side, frame_index] = np.asarray(box_with_conf, dtype=np.float32)

            for frame_index, keypoints_with_conf in track_info["keypoints_with_conf"].items():
                if 0 <= frame_index < length:
                    joints2d[hand_side, frame_index] = np.asarray(keypoints_with_conf, dtype=np.float32)

        np.save(hand_bbox_path, bbox)
        np.save(hand_joints2d_path, joints2d)
