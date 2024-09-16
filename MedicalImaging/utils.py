import torch
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision import models
from torch import nn
import os
from PIL import Image
from config import BATCH_SIZE
import numpy as np
from model import ResNetModel, VGGModel, Eva
class CustomImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        # Convert grayscale images to RGB
        # if img.mode != 'RGB':
        #     img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        label = self.labels[idx]
        return img, label

def load_images_from_folder(folder):
    data = {}
    for label_name in os.listdir(folder):
        label = int(label_name) -1
        images = []
        for filename in os.listdir(os.path.join(folder, label_name)):
            image_path = os.path.join(folder, label_name, filename)
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            images.append(image)
        data[label] = images
    return data

def create_test_dataset(image_folder: str, transform):
    images = []
    image_files = []
    labels = []
    
    # Loop through the subfolders in the "test" folder
    for label_name in os.listdir(image_folder):
        label_folder = os.path.join(image_folder, label_name)
        if os.path.isdir(label_folder):
            # Loop through images in each label's folder
            for img_name in os.listdir(label_folder):
                img_path = os.path.join(label_folder, img_name)
                img = Image.open(img_path).convert("RGB")  # Convert to RGB
                
                images.append(img)
                image_files.append(img_name)
                labels.append(label_name)  # Use folder name as the label

    # Create the CustomImageDataset with the loaded images and their labels
    dataset = CustomImageDataset(images, labels, transform)
    
    return dataset, image_files

def get_model(model_name: str, num_classes: int):
    if model_name == 'resnet50':
        model = ResNetModel(num_classes)
    elif model_name == 'vgg16':
        model = VGGModel(num_classes)
    elif model_name == 'eva':
        model = Eva(num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported")
    return model

def load_images_for_test_data(path: str):
    data = []
    labels = []
    #folders = Covid, Normal, Viral Pneumonia
    for folder in os.listdir(path):
        for filename in os.listdir(os.path.join(path, folder)):
            image_path = os.path.join(path, folder, filename)
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            data.append(image)
            labels.append(folder)
    # labels = np.array(labels)
    # labels = np.where(labels == 'Covid', 2, labels)
    # labels = np.where(labels == 'Normal', 1, labels)
    # labels = np.where(labels == 'Viral Pneumonia', 0, labels)
    labels = np.array(labels)
    labels = labels.astype(int)  # Ensure labels are integers
    return {"data": data, "labels": labels}