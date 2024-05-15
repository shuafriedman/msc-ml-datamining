import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from utils import load_images_from_folder, get_model, CustomImageDataset
from config import *
import utils
from tqdm import tqdm
import pandas as pd

def train_and_eval(dataloaders, num_classes: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name='vgg16', num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    metrics = []  # List to store metrics for each epoch

    max_train_accuracy = 0
    max_validation_accuracy = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_correct = train_total = 0
        running_loss = 0.0
        
        for inputs, labels in tqdm(dataloaders['train'], desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            running_loss += loss.item()

        train_accuracy = 100 * train_correct / train_total
        train_loss = running_loss / len(dataloaders['train'])
        max_train_accuracy = max(max_train_accuracy, train_accuracy)

        validation_accuracy = validate_model(model, dataloaders['test'], device)
        max_validation_accuracy = max(max_validation_accuracy, validation_accuracy)

        # Store metrics
        metrics.append({
            'epoch': epoch + 1,
            'train_accuracy': train_accuracy,
            'max_train_accuracy': max_train_accuracy,
            'validation_accuracy': validation_accuracy,
            'max_validation_accuracy': max_validation_accuracy,
            'train_loss': train_loss
        })

        print(f"Epoch {epoch+1}: Train Acc: {train_accuracy:.2f}%, Max Train Acc: {max_train_accuracy:.2f}%, "
              f"Val Acc: {validation_accuracy:.2f}%, Max Val Acc: {max_validation_accuracy:.2f}%")

    # Convert metrics to DataFrame and save to CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv('training_metrics.csv', index=False)

def validate_model(model, dataloader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    return accuracy

if __name__ == "__main__":
    folds = 2 if not RUN_KFOLD else KFOLDS
    kfold = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    image_dict = load_images_from_folder(DATA_PATH)
    all_images = []
    all_labels = []
    for label, images in image_dict.items():
        all_images.extend(images)
        all_labels.extend([label] * len(images))
    
    for fold, (train_ids, test_ids) in enumerate(kfold.split(all_images)):
        train_images, test_images = [all_images[i] for i in train_ids], [all_images[i] for i in test_ids]
        train_labels, test_labels = [all_labels[i] for i in train_ids], [all_labels[i] for i in test_ids]
        transforms = get_transforms(train_images)  # Assuming this prepares appropriate transforms
        datasets = {
            "train": CustomImageDataset(train_images, train_labels, transform=transforms['train']),
            "test": CustomImageDataset(test_images, test_labels, transform=transforms['test'])
        }
        dataloaders = {
            "train": torch.utils.data.DataLoader(datasets["train"], batch_size=BATCH_SIZE, shuffle=True),
            "test": torch.utils.data.DataLoader(datasets["test"], batch_size=BATCH_SIZE, shuffle=False)
        }
        
        train_and_eval(dataloaders, num_classes=len(image_dict))
