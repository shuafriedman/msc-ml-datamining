import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from utils import load_images_from_folder, get_model
from config import *
import utils
from sklearn.model_selection import KFold

def train_and_eval(dataloaders, num_classes: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name='vgg16', num_classes=num_classes).to(device)    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(5):  # Assuming 10 epochs
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            model.train()
            for data in dataloaders[phase]:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    print(outputs)
                    _, preds = torch.max(outputs, 1)
                    print(preds)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
    
        # save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()})

if __name__ == "__main__":
    folds = 2 if not RUN_KFOLD else KFOLDS
    kfold = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    image_dict = utils.load_images_from_folder(DATA_PATH)
    all_images = []
    all_labels = []
    for label, images in image_dict.items():
        all_images.extend(images)
        all_labels.extend([label] * len(images)) 
    datasets = {}
    dataloaders = {}
    for fold, (train_ids, test_ids) in enumerate(kfold.split(all_images)):
        # Split data into train and test based on the fold
        train_images, test_images = [all_images[i] for i in train_ids], [all_images[i] for i in test_ids]
        train_labels, test_labels = [all_labels[i] for i in train_ids], [all_labels[i] for i in test_ids]

        # Create datasets for this fold
        datasets[f"train"] = utils.CustomImageDataset(train_images, train_labels, transform=TRAIN_TRANSFORM)
        datasets[f"test"] = utils.CustomImageDataset(test_images, test_labels, transform=TEST_TRANSFORM)

        # Create dataloaders for this fold
        dataloaders[f"train"] = torch.utils.data.DataLoader(datasets[f"train"], batch_size=BATCH_SIZE, shuffle=True)
        dataloaders[f"test"] = torch.utils.data.DataLoader(datasets[f"test"], batch_size=BATCH_SIZE, shuffle=False)
        
        train_and_eval(dataloaders, num_classes=len(image_dict))

