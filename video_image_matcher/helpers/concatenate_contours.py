import os
import json
import numpy as np

def concatenate_contours(output_path, character_name, num_frames=None):
    # Load the subject_labels and contours JSON files
    with open(os.path.join(output_path, 'subject_labels.json'), 'r') as f:
        subject_labels = json.load(f)
    
    with open(os.path.join(output_path, 'contours.json'), 'r') as f:
        contours = json.load(f)
    
    # Get the subject IDs for the character
    subject_ids = subject_labels.get(character_name, [])
    
    if not subject_ids:
        raise ValueError(f"No subject IDs found for character '{character_name}'.")

    # Initialize a dictionary to store concatenated contours by frame number
    concatenated_contours = {}

    # Loop through each frame in the contours
    for subject_id in subject_ids:
        subject_key = str(subject_id)  # Convert subject_id to string to match JSON keys
        if subject_key in contours:
            for frame_number_str, frame_contour in contours[subject_key].items():
                frame_number = int(frame_number_str)
                if frame_number not in concatenated_contours:
                    concatenated_contours[frame_number] = []
                concatenated_contours[frame_number].extend(frame_contour)

    # Determine the maximum frame number
    if num_frames is None:
        num_frames = max(concatenated_contours.keys()) if concatenated_contours else 0

    # Create an array where each row corresponds to a frame
    frame_contour_array = np.empty((num_frames + 1,), dtype=object)
    for frame_number in range(num_frames + 1):
        if frame_number in concatenated_contours:
            frame_contour_array[frame_number] = np.array(concatenated_contours[frame_number])
        else:
            frame_contour_array[frame_number] = np.array([])

    return frame_contour_array

