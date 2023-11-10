# train.py

import argparse
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob
from model import get_model
from custom_dataset import get_dataset
import os

def train_model(model, train_data_dir, device, optimizer, criterion, total_epochs=50, save_path="models", model_weight_path="models"):
    train_data = get_dataset(train_data_dir)  # Load the dataset
    train_loader = DataLoader(train_data, batch_size=128, num_workers=6, pin_memory=True)
    scaler = GradScaler()

    # Check if a specific model weight path is provided
    if model_weight_path and os.path.exists(model_weight_path):
        checkpoint = torch.load(model_weight_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        avg_loss_loaded = checkpoint.get('avg_loss', None)
        for g in optimizer.param_groups:
            g['lr'] = args.learning_rate if args.learning_rate is not None else g['lr']
        model.train()
    else:
        # If no specific model weight path is provided, search in save_path
        list_of_files = glob.glob(f'{save_path}/model_epoch_*.pth')
        if list_of_files:  
            latest_file = max(list_of_files, key=os.path.getctime)
            checkpoint = torch.load(latest_file)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            starting_epoch = checkpoint['epoch'] + 1
            train_losses = checkpoint['train_losses']
            avg_loss_loaded = checkpoint.get('avg_loss', None)
            for g in optimizer.param_groups:
                g['lr'] = args.learning_rate if args.learning_rate is not None else g['lr']
            model.train()
        else:
            starting_epoch = 0
            avg_loss_loaded = None
            train_losses = []

    for epoch in range(starting_epoch, total_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}, Learning Rate: {current_lr:.6f}")

        model.train()
        total_loss = 0.0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}", unit="batch") as progress_bar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs.logits, labels)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                progress_bar.set_postfix({"Loss": loss.item()})
                progress_bar.update()

                del inputs, labels, outputs

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_loss': avg_train_loss,
            'scaler_state_dict': scaler.state_dict(),
            'train_losses': train_losses,
        }, f"{save_path}/model_epoch_{epoch}.pth")

        torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model on a dataset.')
    parser.add_argument('train_data_dir', type=str, help='Directory path for training data.')
    parser.add_argument('--total_epochs', type=int, default=50, help='Total number of epochs for training.')
    parser.add_argument('--save_path', type=str, default="/models", help='Path to save the model checkpoints.')
    parser.add_argument('--learning_rate', type=float, default=None, help='Custom learning rate for training. If not provided, uses the learning rate from the checkpoint or a default value.')
    parser.add_argument('--model_weight_path', type=str, default="/models", help='Path to a specific model weight file to resume training. If not provided, will search in the save_path.')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(device)
    lr = args.learning_rate if args.learning_rate is not None else 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-5, eps=1e-8)
    criterion = torch.nn.CrossEntropyLoss()  
    train_model(model, args.train_data_dir, device, optimizer, criterion, args.total_epochs, args.save_path, args.model_weight_path)
