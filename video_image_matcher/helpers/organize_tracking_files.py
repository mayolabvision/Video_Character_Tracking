import os
import shutil

def organize_tracking_files(video_path, output_path, character_name):
    # Extract the base name of the video file (without extension)
    video_base_name = os.path.splitext(os.path.basename(video_path))[0]

    # Create a new directory based on the character name
    tracking_directory = os.path.join(os.path.dirname(video_path),'characterTracking_finalRepo')
    os.makedirs(tracking_directory, exist_ok=True)

    # Define the source files
    video_file = os.path.join(output_path, 'tracked_subject.mp4')
    new_video_file = os.path.join(tracking_directory, f'{video_base_name}_{character_name}_tracking.mp4')
    shutil.copy2(video_file, new_video_file)
    
    csv_file = os.path.join(output_path, f'{character_name}_contours.mat')
    new_csv_file = os.path.join(tracking_directory, f'{video_base_name}_{character_name}_contours.mat')
    shutil.copy2(csv_file, new_csv_file)

    print(f"Files copied and renamed to {tracking_directory}")

