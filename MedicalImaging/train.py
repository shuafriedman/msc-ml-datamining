import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from utils import load_images_from_folder, get_model, CustomImageDataset, load_images_for_test_data
from config import *
import pandas as pd
from tqdm import tqdm

def train_and_eval(model, dataloaders, model_name, fold, lr, num_classes: int, num_epochs: int = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)
    metrics = []

    max_train_accuracy = 0
    max_test_accuracy = 0
    best_epoch = 0
    best_model_state = None
    epochs = num_epochs if num_epochs else NUM_EPOCHS
    for epoch in range(epochs):
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

        test_accuracy = validate_model(model, dataloaders["test"], device)
        if test_accuracy > max_test_accuracy:
            max_test_accuracy = test_accuracy
            best_epoch = epoch + 1
            best_model_state = model.state_dict()
        
        metrics.append({
            'epoch': epoch + 1,
            'train_accuracy': train_accuracy,
            'max_train_accuracy': max_train_accuracy,
            'test_accuracy': test_accuracy,
            'max_test_accuracy': max_test_accuracy,
            'best_epoch': best_epoch,
            'train_loss': train_loss,
            'lr': lr,
            'model_name': model_name,
            'fold': fold + 1
        })
        print(f"{model_name} "
              f"Epoch {epoch+1}: Train Acc: {train_accuracy:.2f}%, Max Train Acc: {max_train_accuracy:.2f}%, "
              f"Test Acc: {test_accuracy:.2f}%, Max Test Acc: {max_test_accuracy:.2f}%, Best Epoch: {best_epoch}")
    metrics_df = pd.DataFrame(metrics)
    return metrics_df, best_model_state

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

def get_config_and_transforms(model):
    if model.config:
        config = model.config["mean"], model.config["std"], (model.config["input_size"][1], model.config["input_size"][2])
    else:
        config = 0.5, 0.5, (224, 224)
        
    return get_transforms(mean=config[0], std=config[1], resize=config[2])

if __name__ == "__main__":
    folds = 2 if not RUN_KFOLD else KFOLDS
    kfold = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    image_dict = load_images_from_folder(DATA_PATH)

    all_images = []
    all_labels = []
    for label, images in image_dict.items():
        all_images.extend(images)
        all_labels.extend([label] * len(images))

    learning_rates = [0.001, 0.0001, 0.00001]
    best_model_state_per_model = {}
    best_lr_per_model = {}
    best_results = []  # To store the best results per model
    all_metrics = []

    for model_name in MODELS:
        print("Loading " + model_name)
        model = get_model(model_name=model_name, num_classes=len(image_dict))
        transforms = get_config_and_transforms(model)

        model_results = []  # Store results for each learning rate across all folds
        
        for fold, (train_ids, test_ids) in enumerate(kfold.split(all_images)):
            print("Testing fold " + str(fold))
            train_images, test_images = [all_images[i] for i in train_ids], [all_images[i] for i in test_ids]
            train_labels, test_labels = [all_labels[i] for i in train_ids], [all_labels[i] for i in test_ids]
            datasets = {
                "train": CustomImageDataset(train_images, train_labels, transform=transforms['train']),
                "test": CustomImageDataset(test_images, test_labels, transform=transforms['test'])
            }
            dataloaders = {
                "train": torch.utils.data.DataLoader(datasets["train"], batch_size=BATCH_SIZE, shuffle=True),
                "test": torch.utils.data.DataLoader(datasets["test"], batch_size=BATCH_SIZE, shuffle=False),
            }

            for lr in learning_rates:
                print(f"Testing lr {lr} for fold {fold + 1}")
                result, model_state = train_and_eval(model, dataloaders, model_name, fold, lr, num_classes=len(image_dict))
                
                # Collect results from each fold for averaging later
                result['fold'] = fold + 1  # Mark fold in the result for tracking
                model_results.append(result)
        
        # Once all folds are done for a model and each learning rate, average the results
        model_results_df = pd.concat(model_results, ignore_index=True)
        avg_results = model_results_df.groupby(['lr', 'model_name']).mean().reset_index()

        # Track the best learning rate and its corresponding results
        best_row = avg_results.loc[avg_results['max_test_accuracy'].idxmax()]
        best_results.append({
            'model_name': model_name,
            'best_lr': best_row['lr'],
            'max_test_accuracy': best_row['max_test_accuracy'],
            'best_epoch': best_row['best_epoch']
        })

        # Save the model with the best learning rate
        best_model_state_per_model[model_name] = model_state
        torch.save(best_model_state_per_model[model_name], f"best_model_{model_name}_lr_{best_row['lr']}.pth")

        # Add the model's metrics to the overall results
        all_metrics.append(avg_results)

    # Save all results into one CSV
    all_metrics_df = pd.concat(all_metrics, ignore_index=True)
    all_metrics_df.to_csv("all_results.csv", index=False)

    # Save only the best models and their parameters
    best_results_df = pd.DataFrame(best_results)
    best_results_df.to_csv("best_models.csv", index=False)