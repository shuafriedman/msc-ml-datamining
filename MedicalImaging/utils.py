import torch
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision import models
from torch import nn
import os
from PIL import Image

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
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
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
    model = eval(f'models.{model_name}(weights=True)')
    for param in model.features.parameters():
        param.required_grad = False
    in_features = model.classifier[-1].in_features
    features = list(model.classifier.children())[:-1] 
    features.extend([nn.Linear(in_features, num_classes)])
    model.classifier = nn.Sequential(*features)     
    return model