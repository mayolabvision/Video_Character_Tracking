import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process a specific video for character tracking.")
    parser.add_argument("video_path", type=str, nargs='?', default=None, help="Path to the specific video file.")
    parser.add_argument("--character_name", type=str, default='Principal', help="Name of character, default is Principal.")
    parser.add_argument("--confidence_threshold", type=float, default=0.6, help="Detection threshold for people in video.")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="Threshold for how dissimilar people need to look to assign new subject ID.")
    parser.add_argument("--max_missing_frames", type=int, default=10, help="Number of frames person can disappear during occlusion before assigning new subject ID.")
    parser.add_argument("--num_contour_points", type=int, default=50, help="Number of contour points used to fit detected person.")
    return parser.parse_args()
