import os
from rembg import remove
from PIL import Image

image_path = '/Users/kendranoneman/Projects/mayo/video_image_matcher/tests/Principal.png'
output_path = '/Users/kendranoneman/Projects/mayo/video_image_matcher/tests/Principal_nobg.png'

def remove_background_image(image_path):
    output_path = os.path.join(os.path.splitext(image_path)[0]+'_noBG'+os.path.splitext(image_path)[1])
    
    if not os.path.exists(output_path):
        input = Image.open(image_path)
        output = remove(input)
        output.save(output_path)
        print(f"Template image created: {output_path}")
    else:
        print("Template image already exists.")
        pass

    return output_path
