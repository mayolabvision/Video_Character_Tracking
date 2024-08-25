import os
import cv2 as cv
from video_image_matcher.helpers import *

image_path = '/Users/kendranoneman/Projects/mayo/video_image_matcher/tests/Principal.png'
video_path = '/Users/kendranoneman/Projects/mayo/video_image_matcher/tests/Exp_Left_FourthDoor_High_v2.mp4' 

def main():
    print('\n\n---------------------------------------- STEP 1. CREATE TEMPLATE IMAGE ----------------------------------------')
    template_image_path = remove_background_image(image_path)
    
    print('\n\n---------------------------------------- STEP 2. LOADING TARGET VIDEO ----------------------------------------')
    person_tracker(video_path)

if __name__ == "__main__":
    main()
