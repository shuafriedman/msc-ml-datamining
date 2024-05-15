import torch
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision import models
from torch import nn
import os
from PIL import Image
from config import BATCH_SIZE
import numpy as np
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
    model = eval(f'models.{model_name}(weights=True)')
    for param in model.features.parameters():
        param.required_grad = False
    #adjust the input of the model to take in greyscale
    # model.features[0] = nn.Conv2d(1, BATCH_SIZE, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    in_features = model.classifier[-1].in_features
    features = list(model.classifier.children())[:-1] 
    # features.extend([nn.Linear(in_features, num_classes)])
    # model.classifier = nn.Sequential(*features)
    #extend with two linear layers
    features.extend([nn.Sequential(
                            nn.Linear(in_features, 256), 
                            nn.ReLU(),
                            nn.Dropout(0.3), 
                            nn.Linear(256, num_classes)
                            )
                    ]
    )
    model.classifier = nn.Sequential(*features)
    return model