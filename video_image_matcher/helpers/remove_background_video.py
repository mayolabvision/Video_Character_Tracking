import cv2 as cv
import os

def remove_background_video(video_path):
    # Extract the directory, filename, and extension from the video_path
    dir_name, base_name = os.path.split(video_path)
    file_name, ext = os.path.splitext(base_name)
    
    # Create the output path by appending '_noBG' to the file name
    output_path = os.path.join(dir_name, f"{file_name}_noBG{ext}")

    # Initialize video capture
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Video at {video_path} could not be opened.")
    
    # Initialize the background subtractor (MOG2) with parameters to retain more foreground
    backSub = cv.createBackgroundSubtractorMOG2(history=1000, varThreshold=100, detectShadows=False)

    # Initialize video writer
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv.CAP_PROP_FPS))
    frame_size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    out = cv.VideoWriter(output_path, fourcc, fps, frame_size, isColor=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply the background subtractor to get the foreground mask
        fg_mask = backSub.apply(frame)

        # Remove small noise by applying a threshold
        _, fg_mask = cv.threshold(fg_mask, 50, 255, cv.THRESH_BINARY)

        # Apply the foreground mask to the original frame
        fg = cv.bitwise_and(frame, frame, mask=fg_mask)

        # Display the resulting frame
        cv.imshow('Foreground', fg)

        # Write the frame to the output video
        out.write(fg)

        # Exit if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv.destroyAllWindows()

# Example usage:
# remove_background('/mnt/data/Exp_Left_FourthDoor_High_v2.mp4')
