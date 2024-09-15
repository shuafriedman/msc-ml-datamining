import torch
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download
from PIL import Image
import os
import logging
import pandas as pd
from torchvision import transforms
from config import TEST_DATA_PATH, HUGGINGFACE_MODELS, BATCH_SIZE
from train import get_config_and_transforms
from utils import get_model, CustomImageDataset
from tqdm import tqdm

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to load a model from Hugging Face
def load_model_from_hf(repo_name: str):
    logging.info(f"Downloading model from Hugging Face: {repo_name}")
    full_repo_name = f"Shuaf98/" + repo_name
    model_path = hf_hub_download(repo_id=full_repo_name, filename="model.pth")
    model_name = repo_name.split("_")[-1]
    model = get_model(model_name, num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    logging.info(f"Model {model_name} loaded successfully.")
    return model

# Updated create_test_dataset function to include folder labels
def create_test_dataset(image_folder: str, transform):
    images = []
    image_files = []
    labels = []

    # Loop through the subfolders in the "test" folder
    for label_name in os.listdir(image_folder):
        label_folder = os.path.join(image_folder, label_name)
        if os.path.isdir(label_folder):
            # Loop through images in each label's folder
            for img_name in os.listdir(label_folder):
                img_path = os.path.join(label_folder, img_name)
                img = Image.open(img_path).convert("RGB")  # Convert to RGB
                
                images.append(img)
                image_files.append(img_name)
                labels.append(label_name)  # Use folder name as the label

    dataset = CustomImageDataset(images, labels, transform)
    return dataset, image_files, labels

# Function to run inference on a batch of images
def run_inference(model, images, device):
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    return predicted.cpu().numpy()  # Return predictions as numpy array

if __name__ == "__main__":
    logging.info("Starting inference script.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load models from Hugging Face
    for model_repo in HUGGINGFACE_MODELS:
        model_name = model_repo.split("_")[-1]
        logging.info(f"Loading model: {model_name} from Hugging Face...")
        model = load_model_from_hf(model_repo)
        model.to(device)

        # Get the transformations for inference
        transforms_dict = get_config_and_transforms(model)

        # Create the test dataset and DataLoader
        logging.info(f"Creating test dataset for model: {model_name}")
        test_dataset, image_files, true_labels = create_test_dataset(TEST_DATA_PATH, transforms_dict["test"])
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        logging.info(f"Test dataset and DataLoader created for model: {model_name}")

        all_predictions = []
        all_true_labels = []
        image_names = []

        # Run inference on the test DataLoader
        for i, (inputs, labels) in tqdm(enumerate(test_loader), desc=f"Running inference for {model_name}"):
            inputs = inputs.to(device)
            predictions = run_inference(model, inputs, device)

            # Store predictions, true labels, and image file names for CSV export
            all_predictions.extend(predictions)
            all_true_labels.extend(labels)
            image_names.extend(image_files[i * BATCH_SIZE: (i + 1) * BATCH_SIZE])

            # Output predictions for each image in the batch
            for batch_idx, img_name in enumerate(image_files[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]):
                logging.info(f"Predictions for {img_name} with {model_name}: {predictions[batch_idx]} (True label: {labels[batch_idx]})")

        # Convert true labels to integer indices for comparison
        correct = 0
        for pred, true_label in zip(all_predictions, all_true_labels):
            if str(pred) == str(true_label):
                correct += 1

        # Calculate accuracy
        total = len(all_predictions)
        accuracy = correct / total * 100 if total > 0 else 0

        logging.info(f"Model {model_name} accuracy: {accuracy:.2f}%")


        # Create a DataFrame to store the image names, predictions, and true labels
        df = pd.DataFrame({
            'Image': image_names,
            'True Label': all_true_labels,
            'Predicted Label': all_predictions
        })

        # Export to CSV
        csv_filename = f"inference_results_{model_name}.csv"
        df.to_csv(csv_filename, index=False)
        logging.info(f"Results saved to {csv_filename}")

    logging.info("Inference script completed.")
