import os
from PIL import Image

base_dir = os.path.expanduser("urban_images")
left_dir = os.path.join("left")
right_dir = os.path.join("right")

os.makedirs(left_dir, exist_ok=True)
os.makedirs(right_dir, exist_ok=True)

for filename in os.listdir(base_dir):
    img_path = os.path.join(base_dir, filename)
    img = Image.open(img_path)

    left_img = img.crop((0, 0, 256, 256))
    right_img = img.crop((256, 0, 512, 256))

    left_img.save(os.path.join(left_dir, filename))
    right_img.save(os.path.join(right_dir, filename))

print("Done!")
