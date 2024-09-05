import os
import csv
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed



# Processing command line arguments
parser = argparse.ArgumentParser(description="Process images to count seeds and save results.")
parser.add_argument('-i', '--input_folder', type=str, default="Seeds", help='Input folder containing images.')
parser.add_argument('-o', '--output_folder', type=str, default="seed_count_image_output", help='Output folder to save processed images.')
parser.add_argument('-c', '--output_csv', type=str, default="seed_count.csv", help='Output CSV file to save seed counts.')

args = parser.parse_args()

# Loading the model in the main thread
model = YOLO('Yolo_segmentation.pt')

# Get all categories
yolo_classes = list(model.names.values())
print(yolo_classes)

# Confidence Threshold
conf = 0.6

# Setting Color
fill_color = [210, 180, 140, 128]  
line_color = [210, 180, 140, 255]  

# Initialize an empty list to store the number of seeds for each image
seed_data = []


os.makedirs(args.output_folder, exist_ok=True)

def process_image(image_name, model):
    img_path = os.path.join(args.input_folder, image_name)

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model.predict(img)

    seed_count = 0

    # Creating a transparent overlay
    overlay = np.zeros((img_rgb.shape[0], img_rgb.shape[1], 4), dtype=np.uint8)
    overlay[..., :3] = img_rgb
    overlay[..., 3] = 128  

    # Processing prediction results
    for result in results:
        for mask, box, class_id, score in zip(result.masks.xy, result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            class_id = int(class_id)
            class_name = yolo_classes[class_id]
            if class_name == "seed" and score > conf:  # Confirm that the category is "seed" and the confidence is greater than the threshold
                seed_count += 1
                points = np.int32([mask])
                # Filling polygons
                cv2.fillPoly(overlay, points, fill_color)
                # Draw polygon edges
                cv2.polylines(overlay, points, isClosed=True, color=line_color, thickness=2)
                # Get the center point of the segmented area
                M = cv2.moments(points)
                if M['m00'] != 0:
                    cX = int(M['m10'] / M['m00'])
                    cY = int(M['m01'] / M['m00'])
                    # Print confidence
                    cv2.putText(overlay, f'{score:.2f}', (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255, 255), 2)

    # Save the image name and number of seeds
    seed_data.append([image_name, seed_count])

    # Print the number of seeds above the image
    cv2.putText(overlay, f'Seed Count: {seed_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255, 255), 2)

    # Convert the original image and overlay to PIL images in RGBA mode
    img_pil = Image.fromarray(img_rgb).convert("RGBA")
    overlay_pil = Image.fromarray(overlay)

    # Merge the overlay with the original image
    combined = Image.alpha_composite(img_pil, overlay_pil)

    # Convert the merged image to RGB mode
    combined = combined.convert("RGB")

    # Saving an image with an overlay
    combined.save(os.path.join(args.output_folder, f"output_{image_name}"))

# Processing images using multithreading
with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(process_image, image_name, model) for image_name in os.listdir(args.input_folder) if image_name.endswith(('.png', '.jpg', '.jpeg'))]
    for future in as_completed(futures):
        future.result()

# Write seed quantity data to a CSV file
with open(args.output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Image Name", "Seed Count"])  
    writer.writerows(seed_data)  

print(f'Seed counts have been saved to {args.output_csv}')
