import torch
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download
from PIL import Image
import os
import logging
from torchvision import transforms
from config import TEST_DATA_PATH, HUGGINGFACE_MODELS, BATCH_SIZE
from train import get_config_and_transforms
from utils import get_model, create_test_dataset
from tqdm import tqdm

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to load a model from Hugging Face
def load_model_from_hf(repo_name: str):
    logging.info(f"Downloading model from Hugging Face: {repo_name}")
    # Download the model file from Hugging Face
    model_path = hf_hub_download(repo_id=repo_name, filename="model.pth")
    
    # Load the model
    model_name = repo_name.split("_")[-1]  # Extract the model type (resnet50, vgg16, eva)
    model = get_model(model_name, num_classes=3)  # Assuming 3 classes for your task
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    logging.info(f"Model {model_name} loaded successfully.")
    return model

# Function to load and preprocess images
def load_and_preprocess_images(image_folder: str, transform):
    logging.info(f"Loading and preprocessing images from {image_folder}")
    images = []
    image_files = []
    
    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)
        img = Image.open(img_path).convert("RGB")  # Convert to RGB
        
        # Apply transformations
        img = transform(img).unsqueeze(0)  # Add batch dimension
        images.append(img)
        image_files.append(img_name)
    
    logging.info(f"Loaded and preprocessed {len(images)} images.")
    return images, image_files

# Function to run inference on an image
def run_inference(model, images, device):
    logging.debug("Running inference on images.")
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    logging.debug("Inference completed.")
    return predicted.cpu().numpy()  # Return predictions as numpy array

if __name__ == "__main__":
    logging.info("Starting inference script.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load models from Hugging Face
    models = {}
    for model_repo in HUGGINGFACE_MODELS:
        model_name = model_repo.split("_")[-1]  # Extract model type
        logging.info(f"Loading model: {model_name} from Hugging Face...")
        models[model_name] = load_model_from_hf(model_repo)
        models[model_name].to(device)

        # Get the transformations for inference (all models use the same transform during inference)
        transforms_dict = get_config_and_transforms(models[model_name])

        # Create the test dataset and DataLoader
        logging.info(f"Creating test dataset for model: {model_name}")
        test_dataset, image_files = create_test_dataset(TEST_DATA_PATH, transforms_dict["test"])
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        logging.info(f"Test dataset and DataLoader created for model: {model_name}")

        # Run inference on the test DataLoader
        for i, (inputs, _) in tqdm(enumerate(test_loader), desc=f"Running inference for {model_name}"):
            logging.info(f"Running inference batch {i+1} for {model_name}")
            inputs = inputs.to(device)  # Send batch to device (GPU or CPU)
            predictions = run_inference(models[model_name], inputs, device)

            # Output predictions for each image in the batch
            batch_start_idx = i * BATCH_SIZE
            for batch_idx, img_name in enumerate(image_files[batch_start_idx: batch_start_idx + BATCH_SIZE]):
                logging.info(f"Predictions for {img_name} with {model_name}: {predictions[batch_idx]}")

    logging.info("Inference script completed.")
