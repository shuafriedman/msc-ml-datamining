import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from utils import load_images_from_folder, get_model, CustomImageDataset, load_images_for_test_data
from config import *
import utils
from tqdm import tqdm
import pandas as pd

def train_and_eval(model, dataloaders, model_name, fold, num_classes: int, lr: float = LEARNING_RATE, num_epochs: int = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)
    metrics = []  # List to store metrics for each epoch

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
            best_model_state = model.state_dict()  # Save the best model state
        
        # Store metrics
        metrics.append({
            'epoch': epoch + 1,
            'train_accuracy': train_accuracy,
            'max_train_accuracy': max_train_accuracy,
            'test_accuracy': test_accuracy,
            'max_test_accuracy': max_test_accuracy,
            'best_epoch': best_epoch,
            'train_loss': train_loss
        })

        print(f"{model_name} Fold {fold+1}/{folds} "
              f"Epoch {epoch+1}: Train Acc: {train_accuracy:.2f}%, Max Train Acc: {max_train_accuracy:.2f}%, "
              f"Test Acc: {test_accuracy:.2f}%, Max Test Acc: {max_test_accuracy:.2f}%, Best Epoch: {best_epoch}")

    # Convert metrics to DataFrame and save to CSV
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

if __name__ == "__main__":
    folds = 2 if not RUN_KFOLD else KFOLDS
    kfold = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    image_dict = load_images_from_folder(DATA_PATH)
    final_test_data = load_images_for_test_data(TEST_DATA_PATH)
    transforms = get_transforms()  # Assuming this prepares appropriate transforms

    final_test_data_dataset = CustomImageDataset(final_test_data["data"], final_test_data["labels"], transform=transforms['test'])

    all_images = []
    all_labels = []
    for label, images in image_dict.items():
        all_images.extend(images)
        all_labels.extend([label] * len(images))
    results = {}
    best_model_name = None
    max_val_acc = 0
    for fold, (train_ids, test_ids) in enumerate(kfold.split(all_images)):
        for model_name in MODELS:
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
            model = get_model(model_name=model_name, num_classes=len(image_dict))
            result, _ = train_and_eval(model, dataloaders,  model_name, fold, num_classes=len(image_dict),)
            result.to_csv(f"results_{model_name}_fold_{fold+1}.csv", index=False)
            results[f"{model_name}_fold_{fold+1}"] = result
            # check if this is the best model
            avg_max_val_acc = result['max_test_accuracy'].mean()
            if avg_max_val_acc > max_val_acc:
                max_val_acc = avg_max_val_acc
                best_model_name = model_name
    print(f"Best model: {best_model_name}, Average max validation accuracy: {max_val_acc}")

    #save the best model on test data
    best_overall_model_state = None
    best_overall_accuracy = 0
    best_lr = 0
    best_epoch = 0
    #model with the highest aaverage max validation accuracy across all folds
    for lr in [0.001, 0.0001, 0.00001]:
        model = get_model(model_name=best_model_name, num_classes=len(image_dict))
        dataloaders = {
            "train": torch.utils.data.DataLoader(CustomImageDataset(all_images, all_labels, transform=transforms['train']), batch_size=BATCH_SIZE, shuffle=True),
            "test": torch.utils.data.DataLoader(final_test_data_dataset, batch_size=BATCH_SIZE, shuffle=False)
        }
        result, best_model_state = train_and_eval(model,  dataloaders, best_model_name, 0, num_classes=len(image_dict), lr=lr, num_epochs=NUM_EPOCHS_FOR_FINAL_MODEL)
        result.to_csv(f"final_results_{best_model_name}_lr_{lr}.csv", index=False)

        if result['max_test_accuracy'].max() > best_overall_accuracy:
            best_overall_accuracy = result['max_test_accuracy'].max()
            best_overall_model_state = best_model_state
            best_lr = lr
            best_epoch = result[result['max_test_accuracy'] == best_overall_accuracy]['epoch'].values[0]

    torch.save(best_overall_model_state, f"best_overall_model.pt")
    #write the best model params to a txt file
    with open("best_overall_model_params.txt", "w") as f:
        f.write(f"learning_rate: {best_lr}\n")
        f.write(f"epoch: {best_epoch}\n")
        f.write(f"accuracy: {best_overall_accuracy}\n")
        
    print(f"Best overall model saved with learning rate {best_lr} and epoch {best_epoch}, achieving accuracy {best_overall_accuracy}%")
    
    
