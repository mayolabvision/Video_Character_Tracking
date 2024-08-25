import cv2 as cv

def read_video(video_path, frame_number=0):
    cap = cv.VideoCapture(video_path)
    fps = int(cap.get(cv.CAP_PROP_FPS))
    num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    cap.set(cv.CAP_PROP_POS_FRAMES, frame_number-1)
    ret, frame = cap.read()

    '''
    while True:
        ch = 0xFF & cv.waitKey(1)
        if ch == 27:
            break   
    '''
    '''
    for thisFrame in range(num_frames):
        ret, frame = cap.read()
        timestamp = (thisFrame + 1) * (1 / fps)
        if not ret:
            print(f"Warning: Frame {thisFrame + 1} could not be read. Exiting.")
            break

        cv.imshow('Target Video', frame)
        cv.waitKey(1)
    '''
    cap.release()
    cv.destroyAllWindows()

    return frame
