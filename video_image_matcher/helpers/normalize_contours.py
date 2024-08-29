import cv2
import numpy as np

def normalize_contours(video_path, contours):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Video at {video_path} could not be opened.")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Close the video file
    cap.release()
    
    # Calculate the center of the frame
    center_x = width / 2
    center_y = height / 2
    
    # Shift and flip the contours so that (0,0) is the center of the frame,
    # with the top-right being (width/2, height/2)
    normalized_contours = contours.copy()
    normalized_contours[:, :, 0] = contours[:, :, 0] - center_x
    normalized_contours[:, :, 1] = center_y - contours[:, :, 1]
    
    return normalized_contours

# Example usage:
# normalized_contours = normalize_contours('path/to/video.mp4', contours)
