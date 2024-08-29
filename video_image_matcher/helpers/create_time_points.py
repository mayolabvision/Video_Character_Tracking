import numpy as np

def create_time_points(num_frames, fps):
    # Calculate the duration of each frame in milliseconds
    frame_duration_ms = 1000 / fps
    
    # Create an array of time points from 0 to (num_frames - 1) multiplied by the frame duration
    time_points_ms = np.arange(num_frames+1) * frame_duration_ms
    
    return time_points_ms.reshape(-1,1)
