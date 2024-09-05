import base64
import io
import json
from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
from transformers import DeiTForImageClassification, DeiTImageProcessor
from ultralytics import YOLO
import cv2
import numpy as np

# Custom transformation class to ensure the image is in RGB format
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

# Load DeiT pretrained model and image processor
model_name = "facebook/deit-base-distilled-patch16-224"
image_processor = DeiTImageProcessor.from_pretrained(model_name)
model = DeiTForImageClassification.from_pretrained(model_name, num_labels=3)

# Load trained model weights
model.load_state_dict(torch.load('Deit_classification.pth'))

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Class names
class_names = ['Flower', 'Pod', 'Seed']

# Load YOLO model
yolo_model = YOLO('Yolo_segmentation.pt')

# Class colors
colors = {
    "flower": ([255, 0, 255, 128], [255, 0, 255, 255]),  # Light purple
    "pod": ([255, 100, 100, 128], [255, 100, 100, 255]),  # Light red
    "seed": ([205, 133, 63, 128], [205, 133, 63, 255])  # Light brown
}

# Create Flask app
app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify_image():
    data = request.json
    if 'image' not in data:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))

        # Preprocess the image
        image = data_transforms(image)
        image = image.unsqueeze(0)  # Add a batch dimension
        image = image.to(device)

        # Perform classification
        with torch.no_grad():
            outputs = model(pixel_values=image).logits
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            max_prob, preds = torch.max(probabilities, 1)
            predicted_class = preds.item()
            confidence = max_prob.item()

        # Return classification result
        if confidence < 0.8:
            predicted_label = "Not Winged Bean Image"
        else:
            predicted_label = class_names[predicted_class]
        
        return jsonify({"class": predicted_label, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/segmentate', methods=['POST'])
def segmentate_image():
    data = request.json
    if 'image' not in data:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Model prediction
        results = yolo_model.predict(img)

        # Get all classes
        yolo_classes = list(yolo_model.names.values())

        # If no segmentation results, return "null"
        if len(results[0].boxes) == 0:
            return jsonify({"segmented_image": "null"})

        # Create transparent overlay
        overlay = np.zeros((img_rgb.shape[0], img_rgb.shape[1], 4), dtype=np.uint8)
        overlay[..., :3] = img_rgb
        overlay[..., 3] = 128  # Add alpha channel

        # Process prediction results
        for result in results:
            for mask, box, class_id in zip(result.masks.xy, result.boxes, result.boxes.cls):
                class_id = int(class_id)
                class_name = yolo_classes[class_id]
                
                if class_name in colors:
                    fill_color, line_color = colors[class_name]
                else:
                    continue  # Skip classes that are not flower, pod, or seed
                
                points = np.int32([mask])
                # Fill polygon
                cv2.fillPoly(overlay, points, fill_color)
                # Draw polygon border
                cv2.polylines(overlay, points, isClosed=True, color=line_color, thickness=2)

        # Convert original image and overlay to RGBA mode in PIL image
        img_pil = Image.fromarray(img_rgb).convert("RGBA")
        overlay_pil = Image.fromarray(overlay)

        # Merge overlay with original image
        combined = Image.alpha_composite(img_pil, overlay_pil)

        # Convert the resulting image from RGBA to RGB
        combined = combined.convert("RGB")

        # Convert the resulting image to base64
        buffered = io.BytesIO()
        combined.save(buffered, format="JPEG")
        result_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({"segmented_image": result_image_base64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/seedcount', methods=['POST'])
def seed_count():
    data = request.json
    if 'image' not in data:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Model prediction
        results = yolo_model.predict(img)

        # Get all classes
        yolo_classes = list(yolo_model.names.values())

        # If no segmentation results, return "null"
        if len(results[0].boxes) == 0:
            return jsonify({"segmented_image": "null","seed_count": 0})

        # Confidence threshold
        conf = 0.6

        # Specify light purple
        fill_color = [205, 133, 63, 128]
        line_color = [205, 133, 63, 255]  # Opaque light purple border

        # Create transparent overlay
        overlay = np.zeros((img_rgb.shape[0], img_rgb.shape[1], 4), dtype=np.uint8)
        overlay[..., :3] = img_rgb
        overlay[..., 3] = 128  # Add alpha channel

        # Count the number of seeds
        seed_count = 0

        # Process prediction results
        for result in results:
            for mask, box, class_id, score in zip(result.masks.xy, result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                class_id = int(class_id)  # Get class ID
                class_name = yolo_classes[class_id]
                if class_name == "seed" and score > conf:  # Confirm class is "seed" and confidence is above threshold
                    seed_count += 1
                    points = np.int32([mask])
                    # Fill polygon
                    cv2.fillPoly(overlay, points, fill_color)
                    # Draw polygon border
                    cv2.polylines(overlay, points, isClosed=True, color=line_color, thickness=2)
                    # Get center of segmented region
                    M = cv2.moments(points)
                    if M['m00'] != 0:
                        cX = int(M['m10'] / M['m00'])
                        cY = int(M['m01'] / M['m00'])
                        # Print confidence score
                        cv2.putText(overlay, f'{score:.2f}', (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255, 255), 2)

        # Convert original image and overlay to RGBA mode in PIL image
        img_pil = Image.fromarray(img_rgb).convert("RGBA")
        overlay_pil = Image.fromarray(overlay)

        # Merge overlay with original image
        combined = Image.alpha_composite(img_pil, overlay_pil)

        # Convert the resulting image from RGBA to RGB
        combined = combined.convert("RGB")

        # Convert the resulting image to base64
        buffered = io.BytesIO()
        combined.save(buffered, format="JPEG")
        result_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({"segmented_image": result_image_base64, "seed_count": seed_count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)
