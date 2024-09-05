import os
import shutil
import torch
import argparse
from PIL import Image
from torchvision import transforms
from transformers import DeiTForImageClassification, DeiTImageProcessor
from concurrent.futures import ThreadPoolExecutor, as_completed

# Custom conversion class to ensure the image is in RGB format
class ConvertToRGB:
    def __call__(self, img):
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        return img

# Image preprocessing
data_transforms = transforms.Compose([
    ConvertToRGB(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Loading the DeiT pre-trained model and image processor
model_name = "facebook/deit-base-distilled-patch16-224"
image_processor = DeiTImageProcessor.from_pretrained(model_name)
model = DeiTForImageClassification.from_pretrained(model_name, num_labels=3)

# Load the trained model weights
model.load_state_dict(torch.load('Deit_classification.pth'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

class_names = ['Flower', 'Pod', 'Seed']

parser = argparse.ArgumentParser(description="Classify images and copy them to respective folders.")
parser.add_argument('-i', '--image_dir', type=str, default='images', help='Directory of images to classify.')
parser.add_argument('-o', '--output_dir', type=str, default='Image_Classification', help='Directory to save classified images.')

args = parser.parse_args()

# Create a destination folder
output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for class_name in class_names:
    class_dir = os.path.join(output_dir, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

def classify_and_copy(image_name):
    image_path = os.path.join(image_dir, image_name)
    if not os.path.isfile(image_path):
        return

    try:
        image = Image.open(image_path)
        image = data_transforms(image)
        image = image.unsqueeze(0)  # Add a batch dimension
        image = image.to(device)

        # Classification
        with torch.no_grad():
            outputs = model(pixel_values=image).logits
            _, preds = torch.max(outputs, 1)
            predicted_class = preds.item()

        # Copy the files to the appropriate folder
        predicted_label = class_names[predicted_class]
        destination_path = os.path.join(output_dir, predicted_label, image_name)
        shutil.copy2(image_path, destination_path)

        print(f'Copied {image_name} to {predicted_label} folder.')
    except Exception as e:
        print(f'Error processing {image_name}: {e}')

# Processing images using multithreading
image_dir = args.image_dir
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(classify_and_copy, image_name) for image_name in os.listdir(image_dir)]
    for future in as_completed(futures):
        future.result()

print('Classification and copying completed.')
