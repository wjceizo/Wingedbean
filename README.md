# Project Structure Overview

This project consists of four main folders: `Classification`, `Segmentation`, `Application`, and `Scripts`. Each folder contains tasks related to the image processing of Winged Bean.

Python Version The scripts in this project are tested and compatible with **Python 3.11.9**. Make sure that your Python environment is set up with this version.

Before running any Python scripts, you need to install the necessary libraries. To do so, run the following command in your terminal:`pip install -r requirements.txt`

## 1. Classification Folder

This folder contains Python scripts for training classification models on the Winged Bean dataset, using the following models:

- **ResNet50**
- **ResNet101**
- **EfficientNet**
- **Swin Transformer** (a Vision Transformer variant)
- **DeiT** (Data-efficient Image Transformer)

Each model's training script is written in Python, allowing users to select either a CNN or ViT model for the classification task. The scripts come with predefined hyperparameters and support custom input arguments for adjusting the training process.

## 2. Segmentation Folder

This folder contains Python scripts for performing image segmentation using two prominent models:

- **Mask R-CNN**
- **YOLOv8**

These models are used to segment annotated Winged Bean datasets. The folder also includes scripts to test the segmentation results, enabling users to evaluate the model's performance on actual data.

## 3. Application Folder

This folder contains a full-stack web application built using Vue.js and Flask, demonstrating image classification, segmentation, and seed counting for Winged Beans:

- The frontend is built with **Vue.js**, and users must run `npm install` and `npm run dev` in the `webview` folder to install dependencies and start the development server.
- The backend is powered by **Flask**. After ensuring the Python environment is set up, users can run `application.py` to start the Flask server.

This application uses a decoupled frontend-backend architecture, allowing users to upload Winged Bean images via a web interface and view the classification and segmentation results.

## 4. Scripts Folder

This folder contains utility scripts for batch processing images, automating classification, segmentation, and seed counting tasks:

- **Classification script (`classification.py`)**: This script allows batch classification of a large number of images, automatically saving the results. Users can specify the input folder using the `-i` parameter and the output folder using the `-o` parameter.
- **Segmentation script (`segmentation.py`)**: This script performs batch segmentation on input images, saving the results by category. Users can specify the input folder using the `-i` parameter and the output folder using the `-o` parameter.
- **Seed Counting script (`seedcount.py`)**: This script uses pre-trained segmentation models to count the seeds in each image automatically. It outputs the results in a CSV file, along with the image paths. The `-i` and `-o` parameters allow users to specify the input and output folders, and the `-c` parameter specifies the path to save the CSV file.

---

## Download Links

Due to file size limitations, the pre-trained models have been uploaded to the following locations. Users can download them and place them in the appropriate folders（Since GitHub cannot upload large files, please use OneDrive to download）:

- GitHub: https://github.com/wjceizo/Wingedbean
- OneDrive: [winged_beans_application](https://uniofnottm-my.sharepoint.com/:f:/g/personal/psxjw18_nottingham_ac_uk/EncFRHmn13VKtC6Pl-l-TeYBqWXjpRfzZvmfxw1D9f3jlw?e=KQJE0q)

---

## Dataset Access

To reproduce the training results, you can obtain the Winged Bean datasets via:

- Contact **Ms. Chong Yuet Tian** for access to the Winged Bean dataset.
- Contact **Professor Tissa Chandesa** for access to the annotated dataset.
