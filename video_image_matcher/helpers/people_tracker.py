import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import numpy as np
from filterpy.kalman import KalmanFilter
import os
import glob
import json

# IOU calculation function
def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate the intersection
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate the union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    # Calculate IOU
    iou = inter_area / union_area
    return iou

# Kalman filter initialization function
def create_kalman_filter(initial_bbox):
    kf = KalmanFilter(dim_x=7, dim_z=4)
    kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                     [0, 1, 0, 0, 0, 1, 0],
                     [0, 0, 1, 0, 0, 0, 1],
                     [0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0, 1]])

    kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0]])

    kf.P *= 10.0  # Lower the uncertainty
    kf.R[2:, 2:] *= 10.0
    kf.Q[-1, -1] *= 0.01
    kf.Q[4:, 4:] *= 0.01

    # Initial state
    x, y, x2, y2 = initial_bbox
    kf.x[:4] = np.array([x, y, x2, y2]).reshape(4, 1)

    return kf

def people_tracker(video_path, output_path, confidence_threshold=0.7, iou_threshold=0.5, max_missing_frames=10):
    # Create a directory to store subject images
    subject_images_path = os.path.join(output_path, "subjects")
    files = glob.glob(os.path.join(subject_images_path, '*'))
    for f in files:
        os.remove(f)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold 
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cpu"  
    
    predictor = DefaultPredictor(cfg)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Video at {video_path} could not be opened.")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    vid_path = os.path.join(output_path, "tracked_allSubjects.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(vid_path, fourcc, fps, (width, height))

    person_id_counter = 0
    tracked_persons = {}
    
    # Dictionary to store contours by frame and person ID
    contour_dict = {} 

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        outputs = predictor(frame)
        instances = outputs["instances"].to("cpu")
        masks = instances.pred_masks.numpy()
        boxes = instances.pred_boxes.tensor.numpy()
        classes = instances.pred_classes.numpy()

        person_indices = [i for i, cls in enumerate(classes) if cls == 0]
        updated_persons = {}
        used_ids = set()

        for i in person_indices:
            mask = masks[i]
            box = boxes[i]
            x1, y1, x2, y2 = map(int, box)

            best_iou = 0
            best_match_id = None

            for person_id, (kf, last_box, missing_frames) in tracked_persons.items():
                if missing_frames > max_missing_frames:
                    continue

                kf.predict()
                predicted_box = kf.x[:4].reshape((4,)).astype(int)
                iou = calculate_iou(predicted_box, box)
                if iou > best_iou and person_id not in used_ids:
                    best_iou = iou
                    best_match_id = person_id

            if best_iou > iou_threshold and best_match_id is not None:
                kf, _, _ = tracked_persons[best_match_id]
                kf.update(np.array([x1, y1, x2, y2]))
                updated_persons[best_match_id] = (kf, box, 0)
                used_ids.add(best_match_id)
            else:
                kf = create_kalman_filter([x1, y1, x2, y2])
                updated_persons[person_id_counter] = (kf, box, 0)
                best_match_id = person_id_counter
                person_id_counter += 1

                # Save the first frame's image of the detected person with a red bounding box
                subject_image = frame.copy()
                cv2.rectangle(subject_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                zoom_out_x1 = max(0, x1 - 20)
                zoom_out_y1 = max(0, y1 - 20)
                zoom_out_x2 = min(width, x2 + 20)
                zoom_out_y2 = min(height, y2 + 20)
                subject_image = subject_image[zoom_out_y1:zoom_out_y2, zoom_out_x1:zoom_out_x2]
                
                subject_image_path = os.path.join(subject_images_path, f"s{best_match_id:03d}.jpg")
                cv2.imwrite(subject_image_path, subject_image)

            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea).squeeze().tolist()

                # Store the contour in the dictionary
                if best_match_id not in contour_dict:
                    contour_dict[best_match_id] = {}
                contour_dict[best_match_id][frame_number] = largest_contour

            colored_mask = np.zeros_like(frame, dtype=np.uint8)
            colored_mask[mask] = (0, 255, 0)  

            frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)

        for person_id in tracked_persons.keys() - updated_persons.keys():
            kf, last_box, missing_frames = tracked_persons[person_id]
            missing_frames += 1
            if missing_frames <= max_missing_frames:
                kf.predict()
                updated_persons[person_id] = (kf, last_box, missing_frames)

        tracked_persons = updated_persons
        out.write(frame)
        cv2.imshow('Person Tracker', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_number += 1

    cap.release()
    out.release()  
    cv2.destroyAllWindows()

    # Save the dictionary to a JSON file
    with open(os.path.join(output_path, "contours.json"), "w") as f:
        json.dump(contour_dict, f)

    print(f"Saved tracked video to {output_path}")
    print(f"Saved contours data to {os.path.join(output_path, 'contours.json')}")

    return num_frames,fps 
