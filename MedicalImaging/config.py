import argparse
import torchvision.transforms as transforms
import torch
def get_args():
    parser = argparse.ArgumentParser(description="Configuration for training and inference")
    parser.add_argument('--data_path', type=str, default='msc-ml-datamining/MedicalImaging/medical_images/Covid19_dataset_project/data', help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and inference')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for data splitting')
    parser.add_argument('--train_size', type=float, default=0.6, help='Training size for data splitting')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs for training')
    parser.add_argument('--run_kfold', type=bool, default=False, help='Run k-fold cross validation')
    parser.add_argument('--k_folds', type=int, default=4, help='Number of folds for k-fold cross validation')
    args = parser.parse_args()
    return args

# Retrieve command line arguments
args = get_args()

# Paths
DATA_PATH = args.data_path
RANDOM_STATE = args.random_state
TRAIN_SIZE = args.train_size
# Hyperparameters
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.num_epochs
LEARNING_RATE = args.learning_rate
RUN_KFOLD = args.run_kfold
KFOLDS = args.k_folds
MODELS = ['resnet50', 'vgg16']
TEST_DATA_PATH = "test"
# Data transformations
def get_transforms():
    # mean = torch.stack([torch.mean(img) for img in images]).mean()
    # std = torch.stack([torch.std(img) for img in images]).mean()
    train_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])

    test_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])
    return {"train": train_transform, "test": test_transform}
