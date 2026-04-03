import os

import cv2
import numpy as np
import torch
from ultralytics import YOLO

current_file_path = os.path.abspath(__file__)
current_folder = os.path.dirname(current_file_path)


def track_hand(image_paths, track_conf=0.55):
    hand_detector_path = os.path.join(current_folder, "hand_detector.pt")
    hand_detector = YOLO(model=hand_detector_path)

    track_result = {}       # {track_id: {frame_index: {box_with_conf: [], hand_side: 0/1}, ...}, ...}

    for frame_index, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)

        with torch.no_grad():
            det_result = hand_detector.track(image, conf=track_conf, persist=True, verbose=False)

            boxes = det_result[0].boxes
            keypoints = det_result[0].keypoints
            
            hand_side_list = boxes.cls.cpu().numpy().tolist()
            
            box_list = boxes.xyxy.cpu().numpy()
            confidence_list = boxes.conf.cpu().numpy()
            box_with_conf_list= np.hstack([box_list, confidence_list[:, None]]).tolist()

            keypoints_with_conf_list = keypoints.data.cpu().numpy().tolist()

            if len(box_with_conf_list) == 0 or len(keypoints_with_conf_list) == 0 or boxes.id is None:
                continue
            
            track_id_list = boxes.id.cpu().numpy().tolist()

            for track_id, box_with_conf, keypoints_with_conf, hand_side in zip(track_id_list, box_with_conf_list, keypoints_with_conf_list, hand_side_list):
                track_id = int(track_id)
                hand_side = int(hand_side)

                if hand_side not in [0, 1]:
                    continue

                if track_id in track_result:
                    if hand_side == track_result[track_id]["hand_side"]:
                        track_result[track_id]["box_with_conf"][frame_index] = box_with_conf
                        track_result[track_id]["keypoints_with_conf"][frame_index] = keypoints_with_conf
                
                else:
                    track_result[track_id] = {
                        "hand_side": hand_side,
                        "box_with_conf": {frame_index: box_with_conf},
                        "keypoints_with_conf": {frame_index: keypoints_with_conf}
                    }

    return track_result

