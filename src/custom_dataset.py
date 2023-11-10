# custom_dataset.py

from torchvision import datasets, transforms
import shutil
import os

def get_transform():
    transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform

def is_valid_file(filename):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
    return filename.endswith(valid_extensions)

def remove_ipynb_checkpoints(data_dir):
    checkpoint_path = os.path.join(data_dir, '.ipynb_checkpoints')
    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)

def get_dataset(data_dir):
    # Remove .ipynb_checkpoints directory if it exists
    remove_ipynb_checkpoints(data_dir)

    transform = get_transform()
    dataset = datasets.ImageFolder(root=data_dir, transform=transform, is_valid_file=is_valid_file)
    return dataset
