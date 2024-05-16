import os
import numpy as np
from PIL import Image

# Prepare directory for saving images
output_folder1 = "image/reconst"
output_folder2 = "image/pred"
os.makedirs(output_folder1, exist_ok=True)
# os.makedirs(output_folder2, exist_ok=True)

# File path
file_path1 = "output/AllReconYFrames.txt"
#file_path2 = "output/AllPredYFrames.txt"
#file_path3 = "output/reconst_GPU.txt"

def process_and_save_images(file_path, output_folder):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    frame_number = 0
    frame_data = []
    for line in lines:
        if line.strip().startswith('Frame'):
            if frame_data:  # If there's data, save the current frame
                image = np.array(frame_data, dtype=np.uint8).reshape((288,352))
                Image.fromarray(image).save(os.path.join(output_folder, f'Frame_{frame_number}.png'))
                frame_data = []  # Reset for next frame
            frame_number += 1
        elif line.strip():  # Collecting pixel values
            frame_data.extend([int(pixel) for pixel in line.split()])

    # Save the last frame
    if frame_data:
        image = np.array(frame_data, dtype=np.uint8).reshape((288,352))
        Image.fromarray(image).save(os.path.join(output_folder, f'Frame_{frame_number}.png'))

    return frame_number, output_folder



# Call the function to process and save images
total_frames, output_path = process_and_save_images(file_path1, output_folder1)
#total_frames, output_path = process_and_save_images(file_path2, output_folder2)
