import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def feature_matching_ORB(template_image_path, source_video_path):
    template = cv.imread(template_image_path, cv.IMREAD_GRAYSCALE)
    if template is None:
        raise ValueError(f"Error: Template image at {template_image_path} could not be loaded.")

    cap = cv.VideoCapture(source_video_path)
    fps = int(cap.get(cv.CAP_PROP_FPS))
    num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        raise ValueError(f"Error: Video at {source_video_path} could not be opened.")

    orb = cv.ORB_create(nfeatures=1000,scoreType=cv.ORB_HARRIS_SCORE)

    kp_template, des_template = orb.detectAndCompute(template, None)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    for thisFrame in range(num_frames):
        ret, frame = cap.read()
        timestamp = (thisFrame + 1) * (1 / fps)
        if not ret:
            print(f"Warning: Frame {thisFrame + 1} could not be read. Exiting.")
            break

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)
        
        i_matches = bf.match(des_template, des_frame)
        i_matches = sorted(i_matches, key=lambda x: x.distance)
        matches = [m for m in i_matches if m.distance < 50]  # Adjust threshold as needed

        # Filter good matches
        if len(matches) > 10:
            # Extract location of good matches
            src_pts = np.float32([kp_template[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Find homography matrix
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

            # Get the bounds of the template in the video frame
            h, w = template.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv.perspectiveTransform(pts, M)

            # Draw bounding box
            frame = cv.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv.LINE_AA)

        # Display the frame with the bounding box
        cv.imshow('Frame', frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

