# main.py

import argparse
import torch
from PIL import Image
from model import get_model 
from custom_dataset import get_transform 
import glob
import os

def predict_single_image(image_path, model, device, transform):
    # Load and transform the image
    image = Image.open(image_path).convert("RGB")
    transformed_image = transform(image).unsqueeze(0).to(device)

    # Make a prediction
    model.eval()
    with torch.no_grad():
        outputs = model(transformed_image)
        _, predicted = outputs.logits.max(1)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

    # Map numeric labels to string labels
    label_map = {0: "real", 1: "fake"}
    predicted_label = label_map[predicted.item()]

    return predicted_label, probabilities

def load_latest_model(model, device, weights_folder):
    # Use the specified folder or the default path to find the latest model
    list_of_files = glob.glob(os.path.join(weights_folder, 'model_epoch_*.pth'))
    if not list_of_files:
        raise FileNotFoundError(f"No model files found in {weights_folder}.")
    latest_file = max(list_of_files, key=os.path.getctime)
    checkpoint = torch.load(latest_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def main(image_path, weights_folder):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(device)
    model = load_latest_model(model, device, weights_folder)

    transform = get_transform()
    predicted_label, probabilities = predict_single_image(image_path, model, device, transform)
    
    print(f'Predicted label: {predicted_label}')
    print(f'Class probabilities: {probabilities}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict an image using a trained model.')
    parser.add_argument('image_path', type=str, help='Path to the image file for prediction.')
    parser.add_argument('--weights_folder', type=str, default='./models', help='Folder path for model weights. Defaults to ./models if not provided.')

    args = parser.parse_args()
    main(args.image_path, args.weights_folder)
