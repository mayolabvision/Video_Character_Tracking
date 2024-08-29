import cv2
import numpy as np
import os

def highlight_video_with_contours(video_path, output_path, contours_3d):
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Video at {video_path} could not be opened.")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Prepare the output video writer
    output_video_path = os.path.join(output_path, 'tracked_subject.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Check if the current frame has valid contours
        if frame_number < contours_3d.shape[0]:
            contour = contours_3d[frame_number]
            if not np.isnan(contour).all():
                # Convert the contour to a format that cv2 can draw
                contour = contour.astype(np.int32).reshape((-1, 1, 2))
    
                # Draw the contour on the frame
                cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow('Video with Contours', frame)
        
        # Write the frame to the output video
        out.write(frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_number += 1

    cap.release()
    out.release()  # Save the video
    cv2.destroyAllWindows()

    print(f"Saved video with contours to {output_video_path}")

# Example usage:
# contours_3d = np.load('path_to_contours_3d.npy')  # Assuming contours_3d is saved as a .npy file
# highlight_video_with_contours('/path/to/video.mp4', '/path/to/output_directory', contours_3d)
