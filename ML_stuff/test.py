import pandas as pd
from ultralytics import YOLO
import torch
from torch import library, utils, hub
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
# from utils import save_model
import os
import csv
from pathlib import Path
from custom_dataloader import groceries_DataLoader
from train_function import train_data


from typing import Tuple, Dict, List

torch.manual_seed(420)

# Set device agnostic code  
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda': torch.cuda.empty_cache()

# # Generating file list
# path = '/Users/mralw/Desktop/MPHY0049/ML_Stuff/GroceryStoreDataset-master/dataset/train'
# file_list = []

# for root, dirs, files in os.walk(path):
#     for file in files:
#         file_list.append(os.path.join(root, file))
   
# # generating folder list
# path = '/Users/mralw/Desktop/MPHY0049/ML_Stuff/GroceryStoreDataset-master/dataset/train'
# folder_list = []

# for root, dirs, files in os.walk(path):
#     if len(dirs) > 0:
#         folder_list.append(dirs)

## Load the pretrained model
from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n-cls.pt')


# I dont think I've loaded the model correctly 

MODELS_FOLDER = Path("models")
MODEL_SAVE_PATH = MODELS_FOLDER / "groceries.pth" # Importing based on the path name

### Groceries Model 

# Get path to grocery folders
image_path = Path('/Users/mralw/Desktop/MPHY0049/ML_Stuff/GroceryStoreDataset-master/dataset')
train_path = image_path / 'train'
test_path = image_path / 'val'

# get class names from csv
class_names = []
with open(image_path / 'classes.csv', 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip the first row
    for row in csv_reader:
        class_names.append(row[0])

# Instantiate the transforms
transformSequence = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])

# batch_size and num_workers
BATCH_SIZE =  64
NUM_WORKERS = os.cpu_count()

# Training Dataset
datasetTrain, datasetTest, class_names = groceries_DataLoader(
    train_dir = train_path,
    test_dir = test_path,
    transform = transformSequence,
    BATCH_SIZE = BATCH_SIZE,
    NUM_WORKERS = NUM_WORKERS)

# Adjust the fully connected layer to output only three classes
output_shape = len(class_names)
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(p=0.02, inplace=True),
    torch.nn.Linear(in_features=2048,
                    out_features=output_shape,
                    bias = True).to(device)
    )

# Set number of epochs
NUM_EPOCHS = 90
lr = 0.001


# Setup loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params=model.parameters(),
                                lr=lr)


# Setup training loop
results = train_data(model = model,
                train_dataloader = datasetTrain,
                test_dataloader = datasetTest,
                MODEL_SAVE_PATH = MODEL_SAVE_PATH,
                optimizer = optimizer,
                loss_fn = loss_fn,
                epochs = NUM_EPOCHS,
                device = device)

## it's loading but it isn't initiating the training loop

# save_model(model=model,
#            target_dir='models',
#            model_name="1_cheXpert_15_epoch_VGG13_BN_6-Classes.pth")  # Change the model name with new experiment