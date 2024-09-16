import torch
import torch.optim as optim
from utils import load_images_from_folder, get_model, CustomImageDataset, load_images_for_test_data
from config import *
import pandas as pd
from tqdm import tqdm

def train_and_eval(model, dataloaders, model_name, lr, param_size, num_classes: int, num_epochs: int = None):
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
            'param_size': param_size,
            'model_name': model_name
        })
        print(f"{model_name} "
              f"LR: {lr}, Param Size: {param_size}, "
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
    print("Loading in images for train")
    image_dict = load_images_from_folder(DATA_PATH)
    print("Loading in images for test")
    final_test_data = load_images_for_test_data(TEST_DATA_PATH)

    all_images = []
    all_labels = []
    for label, images in image_dict.items():
        all_images.extend(images)
        all_labels.extend([label] * len(images))

    learning_rates = [0.001, 0.0001]
    param_sizes = [128, 256]
    results = {}
    best_model_state_per_model = {}
    best_lr_per_model = {}
    best_param_size_per_model = {}
    best_results = []  # To store the best results per model
    all_metrics = []

    for model_name in MODELS:
        print(f"Getting Model: {model_name}")
        model = get_model(model_name=model_name, num_classes=len(image_dict))
        print("Getting transformations")
        transforms = get_config_and_transforms(model)

        # Create dataloaders for training and test data
        dataloaders = {
            "train": torch.utils.data.DataLoader(
                CustomImageDataset(all_images, all_labels, transform=transforms['train']), 
                batch_size=BATCH_SIZE, shuffle=True
            ),
            "test": torch.utils.data.DataLoader(
                CustomImageDataset(final_test_data["data"], final_test_data["labels"], transform=transforms['test']), 
                batch_size=BATCH_SIZE, shuffle=False
            )
        }

        best_overall_test_accuracy = 0 
        best_model_state = None
        best_lr = None
        best_param_size = None
        best_result = None

        # Loop over learning rates
        for lr in learning_rates:
            for param_size in param_sizes:
                print(f"Starting training for lr: {lr} and param_size: {param_size}")
                # Reinitialize the model for each learning rate
                model = get_model(model_name=model_name, num_classes=len(image_dict))
                result, model_state = train_and_eval(model, dataloaders, model_name, lr, param_size, num_classes=len(image_dict))
                all_metrics.append(result)

                # Get the maximum test accuracy achieved with this learning rate
                current_test_accuracy = result['test_accuracy'].iloc[-1]  # Test accuracy of the last epoch
                # If the current test accuracy is the highest so far, update the best model information
                if current_test_accuracy > best_overall_test_accuracy:                
                    best_overall_test_accuracy = current_test_accuracy
                    best_model_state = model_state
                    best_lr = lr
                    best_param_size = param_size
                    best_result = result

        # Save the best model state and related information
        best_model_state_per_model[model_name] = best_model_state
        best_lr_per_model[model_name] = best_lr
        best_param_size_per_model[model_name] = best_param_size
        best_results.append({
            'model_name': model_name,
            'best_lr': best_lr,
            'best_param_size': best_param_size,
            'max_test_accuracy': best_overall_test_accuracy,
            'best_epoch': best_result['best_epoch'].max()
        })
        torch.save(best_model_state, f"test_results/best_model_{model_name}.pth")


    # Save all combined results
    all_metrics_df = pd.concat(all_metrics, ignore_index=True)
    all_metrics_df.to_csv("test_results/all_results.csv", index=False)

    # Save only the best models and their parameters
    best_results_df = pd.DataFrame(best_results)
    best_results_df.to_csv("test_results/best_models.csv", index=False)
