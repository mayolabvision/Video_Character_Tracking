import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from scipy.spatial import distance
import numpy as np
from filterpy.kalman import KalmanFilter
import os

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

def person_tracker(video_path):
    # Load Detectron2 model configuration and weights
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9  # Set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cpu"  # Run on CPU since we are not using CUDA
    
    predictor = DefaultPredictor(cfg)
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Video at {video_path} could not be opened.")
    
    # Get the original video's properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Set up the video writer to save the output video
    output_path = os.path.splitext(video_path)[0] + "_tracked.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    person_id_counter = 0
    tracked_persons = {}

    max_missing_frames = 50  # Max frames to retain an ID for a missing person

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run object detection on the current frame
        outputs = predictor(frame)
        boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        classes = outputs["instances"].pred_classes.cpu().numpy()

        # Filter for "person" class (class_id == 0 for COCO dataset)
        person_boxes = [box for box, cls in zip(boxes, classes) if cls == 0]

        updated_persons = {}
        used_ids = set()

        for box in person_boxes:
            x1, y1, x2, y2 = map(int, box)

            best_iou = 0
            best_match_id = None

            for person_id, (kf, last_box, missing_frames) in tracked_persons.items():
                if missing_frames > max_missing_frames:
                    continue

                # Predict the next position using Kalman Filter
                kf.predict()
                predicted_box = kf.x[:4].reshape((4,)).astype(int)
                iou = calculate_iou(predicted_box, box)
                if iou > best_iou and person_id not in used_ids:
                    best_iou = iou
                    best_match_id = person_id

            if best_iou > 0.3 and best_match_id is not None:
                kf, _, _ = tracked_persons[best_match_id]
                kf.update(np.array([x1, y1, x2, y2]))
                updated_persons[best_match_id] = (kf, box, 0)
                used_ids.add(best_match_id)
            else:
                kf = create_kalman_filter([x1, y1, x2, y2])
                updated_persons[person_id_counter] = (kf, box, 0)
                person_id_counter += 1

        # Update the missing frame count for persons not detected in the current frame
        for person_id in tracked_persons.keys() - updated_persons.keys():
            kf, last_box, missing_frames = tracked_persons[person_id]
            missing_frames += 1
            if missing_frames <= max_missing_frames:
                # Predict the next position for the missing person
                kf.predict()
                updated_persons[person_id] = (kf, last_box, missing_frames)

        # Draw bounding boxes with person IDs
        for person_id, (kf, box, missing_frames) in updated_persons.items():
            if missing_frames == 0:  # Only draw if the person is detected or within the grace period
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Person {person_id}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        tracked_persons = updated_persons

        # Write the frame to the output video
        out.write(frame)

        # Display the frame with detections
        cv2.imshow('Person Tracker', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()  # Save the video file
    cv2.destroyAllWindows()

    print(f"Saved tracked video to {output_path}")

# Example usage:
# person_tracker('path/to/video.mp4')
