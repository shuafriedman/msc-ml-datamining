import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from utils import load_images_from_folder, get_model, CustomImageDataset
from config import *
import utils
from tqdm import tqdm
import pandas as pd

def train_and_eval(model, dataloaders, num_classes: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model = model.to(device)
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
        epoch_with_max_val = epoch if validation_accuracy == max_validation_accuracy else epoch_with_max_val
        # Store metrics
        metrics.append({
            'epoch': epoch + 1,
            'train_accuracy': train_accuracy,
            'max_train_accuracy': max_train_accuracy,
            'validation_accuracy': validation_accuracy,
            'max_validation_accuracy': max_validation_accuracy,
            'epoch_with_max_val': epoch_with_max_val,
            'train_loss': train_loss
        })

        print(f"Epoch {epoch+1}: Train Acc: {train_accuracy:.2f}%, Max Train Acc: {max_train_accuracy:.2f}%, "
              f"Val Acc: {validation_accuracy:.2f}%, Max Val Acc: {max_validation_accuracy:.2f}%")

    # Convert metrics to DataFrame and save to CSV
    metrics_df = pd.DataFrame(metrics)
    return metrics_df

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
    results = {}
    for fold, (train_ids, test_ids) in enumerate(kfold.split(all_images)):
        for model_name in MODELS:
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
            model = get_model(model_name=model_name, num_classes=len(image_dict))
            result = train_and_eval(model, dataloaders, num_classes=len(image_dict))
            result.to_csv(f"results_{model_name}_fold_{fold}.csv", index=False)
            results[f"{model_name}_fold_{fold}"] = result
    #model with the highest aaverage max validation accuracy across all folds
    max_val_acc = 0
    best_model = None
    for model_name in MODELS:
        avg_max_val_acc = sum([result['max_validation_accuracy'].max() for result in results.values()]) / len(results)
        if avg_max_val_acc > max_val_acc:
            max_val_acc = avg_max_val_acc
            best_model = model_name
    print(f"Best model: {best_model}, Average max validation accuracy: {max_val_acc}")
    # #results summary, create several metrics that are useful
    # summary = pd.DataFrame(results)
    # summary.to_csv("results_summary.csv", index=False)
    
