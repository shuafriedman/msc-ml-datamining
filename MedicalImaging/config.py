import argparse
import torchvision.transforms as transforms

def get_args():
    parser = argparse.ArgumentParser(description="Configuration for training and inference")
    parser.add_argument('--data_path', type=str, default='msc-ml-datamining/MedicalImaging/medical_images/Covid19_dataset_project/data', help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and inference')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for data splitting')
    parser.add_argument('--train_size', type=float, default=0.8, help='Training size for data splitting')
    parser.add_argument('--run_kfold', type=bool, default=False, help='Run k-fold cross validation')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds for k-fold cross validation')
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
LEARNING_RATE = args.learning_rate
RUN_KFOLD = args.run_kfold
KFOLDS = args.k_folds
# Data transformations
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
