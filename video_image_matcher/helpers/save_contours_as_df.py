import os
import numpy as np
import pandas as pd
import scipy.io as sio

def save_contours_as_df(video_path, output_path, character_name, contours, time_points):
    # Extract video name without path and file extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Prepare the data dictionary for each row
    num_frames = contours.shape[0]
    data = {
        'contours': [contours[i] for i in range(num_frames)],  # List of 2D arrays for contours
        'time_points': time_points.flatten(),  # Flattened time points array
        'video_name': [video_name] * num_frames,  # Video name repeated for each frame
        'character_name': [character_name] * num_frames  # Character name repeated for each frame
    }
    
    # Convert the dictionary into a pandas DataFrame
    df = pd.DataFrame(data)
    
    # Convert the DataFrame to a dictionary of arrays for saving to .mat
    mat_dict = {
        'video_name': df['video_name'].values,
        'character_name': df['character_name'].values,
        'contours': np.array(df['contours'].tolist(), dtype=object),
        'time_points': df['time_points'].values
    }
    
    # Save the dictionary to a .mat file
    mat_path = os.path.join(output_path, f'{character_name}_contours.mat')
    sio.savemat(mat_path, mat_dict)
    
    print(f"DataFrame saved as .mat at {mat_path}")

