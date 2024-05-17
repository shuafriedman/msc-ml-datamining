import torch
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision import models
from torch import nn
import os
from PIL import Image
from config import BATCH_SIZE
import numpy as np
from model import ResNetModel, VGGModel
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

def load_images_for_inference(path: str):
    images = []
    for filename in os.listdir(path):
        image_path = os.path.join(path, filename)
        image = Image.open(image_path)
        images.append(image)
    return images

def get_model(model_name: str, num_classes: int):
    if model_name == 'resnet50':
        model = ResNetModel(num_classes)
    elif model_name == 'vgg16':
        model = VGGModel(num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported")
    return model