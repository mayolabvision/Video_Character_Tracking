import os

def create_video_directory(video_path):
    # Extract the directory and filename without extension
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Create the new directory path
    new_directory_path = os.path.join(video_dir, video_name)

    # Create the directory if it doesn't exist
    if not os.path.exists(new_directory_path):
        os.makedirs(os.path.join(new_directory_path,'subjects'))
        print(f"Directory '{new_directory_path}' created.")
    else:
        print(f"Directory '{new_directory_path}' already exists.")

    return new_directory_path
