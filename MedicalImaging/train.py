import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from utils import load_images_from_folder, get_model, CustomImageDataset
from config import *
import utils
from tqdm import tqdm

def train_and_eval(dataloaders, num_classes: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name='vgg16', num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        # Set model to training mode
        model.train()
        train_loader = tqdm(dataloaders['train'], desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} Training")
        running_loss = 0.0
        correct = total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Statistics
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        # Calculate training accuracy and average loss
        train_accuracy = 100 * correct / total
        train_loss = running_loss / len(dataloaders['train'])
        
        # Validate on training and validation data
        # train_val_accuracy = validate_model(model, dataloaders['train'], device, "Train")
        val_accuracy = validate_model(model, dataloaders['test'], device, "Validation")

        # Display results
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] - '
              f'Train Loss: {train_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.2f}%, '
            #   f'Train Validation Accuracy: {train_val_accuracy:.2f}%, '
              f'Validation Accuracy: {val_accuracy:.2f}%')

def validate_model(model, dataloader, device, mode="Validation"):
    model.eval()  # Set model to evaluation mode
    val_loader = tqdm(dataloader, desc=f"{mode} Accuracy Calculation")
    correct = total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    val_loader.close()
    return accuracy

if __name__ == "__main__":
    folds = 4 if not RUN_KFOLD else KFOLDS
    kfold = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    image_dict = load_images_from_folder(DATA_PATH)
    all_images = []
    all_labels = []
    for label, images in image_dict.items():
        all_images.extend(images)
        all_labels.extend([label] * len(images))
    
    for fold, (train_ids, test_ids) in enumerate(kfold.split(all_images)):
        train_images, test_images = [all_images[i] for i in train_ids], [all_images[i] for i in test_ids]
        print(len(train_images), len(test_images))
        train_labels, test_labels = [all_labels[i] for i in train_ids], [all_labels[i] for i in test_ids]
        transforms = get_transforms(train_images)
        datasets = {
            "train": CustomImageDataset(train_images, train_labels, transform=transforms['train']),
            "test": CustomImageDataset(test_images, test_labels, transform=transforms['test'])
        }
        dataloaders = {
            "train": torch.utils.data.DataLoader(datasets["train"], batch_size=BATCH_SIZE, shuffle=True),
            "test": torch.utils.data.DataLoader(datasets["test"], batch_size=BATCH_SIZE, shuffle=False)
        }
        
        train_and_eval(dataloaders, num_classes=len(image_dict))
