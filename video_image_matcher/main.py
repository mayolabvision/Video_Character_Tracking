import os
import cv2 as cv
from video_image_matcher.helpers import *
import time
import signal

def main():
    args = parse_arguments()

    print('\n\n---------------------------------------- STEP 1. CREATE OUTPUT DIRECTORY ----------------------------------------')
    output_path = create_video_directory(args.video_path)
    params = load_params(output_path)
        
    params['character_name'] = args.character_name
    params['video_path'] = args.video_path

    print('\n\n---------------------------------------- STEP 2. TRACK PEOPLE IN VIDEO ----------------------------------------')
    if not os.path.exists(os.path.join(output_path,'tracked_allSubjects.mp4')):
        print('Tracking all subjects')
        num_frames,fps = people_tracker(args.video_path, output_path, confidence_threshold=args.confidence_threshold, iou_threshold=args.iou_threshold, max_missing_frames=args.max_missing_frames)
        params['confidence_threshold'] = args.confidence_threshold
        params['iou_threshold'] = args.iou_threshold
        params['max_missing_frames'] = args.max_missing_frames
        params['sampling_rate'] = fps
        params['num_frames'] = num_frames
        save_params(params, output_path)
    else:
        overwrite = prompt_user_to_overwrite('tracked_allSubjects.mp4 already exists. Do you want to overwrite it? (y/n): ')
        if overwrite:
            print('Tracking all subjects')
            num_frames,fps = people_tracker(args.video_path, output_path, confidence_threshold=args.confidence_threshold, iou_threshold=args.iou_threshold, max_missing_frames=args.max_missing_frames)
            params['confidence_threshold'] = args.confidence_threshold
            params['iou_threshold'] = args.iou_threshold
            params['max_missing_frames'] = args.max_missing_frames
            params['sampling_rate'] = fps
            params['num_frames'] = num_frames
            save_params(params, output_path)
        else:
            print('Skipping tracking all subjects')
            pass
   
    time_points = create_time_points(params['num_frames'],params['sampling_rate'])

    print('\n\n---------------------------------------- STEP 3. TRACK SPECIFIC CHARACTER ----------------------------------------')
    overwrite = prompt_user_to_overwrite('Have you set the subject labels yet? (y/n): ')

    if overwrite:
        raw_contours = concatenate_contours(output_path, args.character_name, num_frames=params['num_frames'])
        contours_3d = simplify_contours(raw_contours, num_points=args.num_contour_points, epsilon_factor=0.01)
        interp_contours = interpolate_contours(contours_3d,output_path,args.character_name)

        params['num_contour_points'] = args.num_contour_points
        save_params(params, output_path)
        
        highlight_video_with_contours(args.video_path, output_path, interp_contours)
        contours = normalize_contours(args.video_path, interp_contours)
    else:
        print('Set the labels for the subject and then come back.')
        pass

    if overwrite:
        print('\n\n---------------------------------------- STEP 4. APPEND RESULTS TO MASTER TABLE ----------------------------------------')
        print('\n')
        save_contours_as_df(args.video_path, output_path, args.character_name, contours, time_points)
        organize_tracking_files(args.video_path, output_path, args.character_name)        
        #update_master_table(args.video_path, args.character_name, contours, time_points)
    

    print('\n\n ============================================================================================\n\n')

if __name__ == "__main__":
    main()
