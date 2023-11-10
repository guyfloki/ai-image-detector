# evaluate.py

import argparse
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import glob
import os
from model import get_model
from custom_dataset import get_dataset

def load_latest_model(model, device, weights_folder):
    # Use the specified folder or the default path to find the latest model
    list_of_files = glob.glob(os.path.join(weights_folder, 'model_epoch_*.pth'))
    if not list_of_files:
        raise FileNotFoundError(f"No model files found in {weights_folder}.")
    latest_file = max(list_of_files, key=os.path.getctime)
    checkpoint = torch.load(latest_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def evaluate_model(test_data_dir, device, criterion, weights_folder):
    test_data = get_dataset(test_data_dir)  # Load the dataset
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=6, pin_memory=True)
    model = get_model(device)
    model = load_latest_model(model, device, weights_folder)
    model.eval()
    test_loss = 0
    all_predicted = []
    all_labels = []

    with torch.no_grad():
        with tqdm(total=len(test_loader), desc="Evaluation", unit="batch") as progress_bar:
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs.logits, labels)

                test_loss += loss.item()
                _, predicted = outputs.logits.max(1)
                all_predicted.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                progress_bar.set_postfix({"Test Loss": loss.item()})
                progress_bar.update()

    all_labels = np.array(all_labels)
    all_predicted = np.array(all_predicted)
    avg_test_loss = test_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_predicted)
    precision = precision_score(all_labels, all_predicted, average='macro')
    recall = recall_score(all_labels, all_predicted, average='macro')
    f1 = f1_score(all_labels, all_predicted, average='macro')

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predicted)
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_confusion_matrix(conf_mat=cm, figsize=(10, 10), show_absolute=True, show_normed=True)
    plt.show()
    plt.savefig("/metrics/Custom_Evaluation.png")
    return avg_test_loss, accuracy, precision, recall, f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained model on a dataset.')
    parser.add_argument('test_data_dir', type=str, help='Directory path for test data.')
    parser.add_argument('--weights_folder', type=str, default='./models', help='Folder path for model weights. Defaults to ./models if not provided.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()  

    # Capture the returned values from the evaluate_model function
    avg_test_loss, accuracy, precision, recall, f1 = evaluate_model(args.test_data_dir, device, criterion, args.weights_folder)

    # Print out the captured metrics
    print(f"Average Test Loss: {avg_test_loss}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
