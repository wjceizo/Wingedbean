import os
import argparse
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setting the default confidence threshold
conf = 0.6

# Defining Colors
colors = {
    "flower": ([255, 0, 255, 128], [255, 0, 255, 255]),  # Light purple
    "pod": ([255, 100, 100, 128], [255, 100, 100, 255]),  # Light red
    "seed": ([205, 133, 63, 128], [205, 133, 63, 255])  # Light brown
}

# Processing command line arguments
parser = argparse.ArgumentParser(description="Batch process images for segmentation.")
parser.add_argument('-i', '--input_dir', type=str, default='images', help='Directory of images to process.')
parser.add_argument('-o', '--output_dir', type=str, default='Image_Segmentation', help='Directory to save processed images.')

args = parser.parse_args()

# Loading the model
model = YOLO('Yolo_segmentation.pt')

# Get all categories
yolo_classes = list(model.names.values())

os.makedirs(args.output_dir, exist_ok=True)
output_image_dir = os.path.join(args.output_dir, 'output_image')
os.makedirs(output_image_dir, exist_ok=True)

for class_name in colors.keys():
    class_dir = os.path.join(args.output_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

def process_image(image_name):
    image_path = os.path.join(args.input_dir, image_name)

    # Check if it is an image file
    if not os.path.isfile(image_path) or not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        return

    # Reading an Image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Model predictions
    results = model.predict(img)

    # Creating a transparent overlay
    overlay = np.zeros((img_rgb.shape[0], img_rgb.shape[1], 4), dtype=np.uint8)
    overlay[..., :3] = img_rgb
    overlay[..., 3] = 128  

    # Processing prediction results
    for result in results:
        for mask, box, class_id in zip(result.masks.xy, result.boxes, result.boxes.cls):
            class_id = int(class_id)
            class_name = yolo_classes[class_id]

            if class_name in colors:
                fill_color, line_color = colors[class_name]
            else:
                continue  

            points = np.int32([mask])
            # Filling polygons
            cv2.fillPoly(overlay, points, fill_color)
            # Draw polygon edges
            cv2.polylines(overlay, points, isClosed=True, color=line_color, thickness=2)

            # Extract bounding boxes of segmented regions
            x, y, w, h = cv2.boundingRect(points)
            cropped_part = img_rgb[y:y+h, x:x+w]

            # Creating an image with a transparent background
            segmented_rgba = np.zeros((h, w, 4), dtype=np.uint8)

            # Copy the segmented portion of the original image to the transparent background image
            for c in range(3):
                segmented_rgba[..., c] = cropped_part[..., c]

            alpha_channel = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(alpha_channel, [points - [x, y]], 255)
            segmented_rgba[..., 3] = alpha_channel

            # Generate unique file names
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            part_output_path = os.path.join(args.output_dir, class_name, f'{class_name}_{timestamp}.png')
            Image.fromarray(segmented_rgba).save(part_output_path)

    # Convert the original image and overlay to PIL images in RGBA mode
    img_pil = Image.fromarray(img_rgb).convert("RGBA")
    overlay_pil = Image.fromarray(overlay)

    # Merge the overlay with the original image
    combined = Image.alpha_composite(img_pil, overlay_pil)

    # Save the image to the output_image folder
    output_image_name = f'segmented_{os.path.splitext(image_name)[0]}.png'
    output_path = os.path.join(output_image_dir, output_image_name)
    combined.save(output_path)

    print(f'Processed and saved {image_name} as {output_image_name}.')

# Processing images using multithreading
with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(process_image, image_name) for image_name in os.listdir(args.input_dir)]
    for future in as_completed(futures):
        future.result()

print('Batch processing completed.')
