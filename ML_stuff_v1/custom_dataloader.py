"""
This is a separate python file containing all functions essential for creating dataloder
"""

import numpy as np
import pandas as pd
import csv

import torch
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets

from pathlib import Path

from PIL import Image

def groceries_DataLoader(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    BATCH_SIZE: int,
    NUM_WORKERS: int
):
    """
    Creates an instance of a dataset (most likely application in: test, train, val)

    Takes image paths and labels we are interested in and returns them as an iterable
    entity. It also applies transforms to images.

    Args:
    data_PATH: path to csv file with information about the images
    class_names: pathologies we would like to export from initial dataset -> keep at 6!
    policy: what to do with uncertain (labelled '-1') data
    multilabel: whether to allow multilabelled images into the dataset
    transform: sequence of transforms that you want applied to the dataset

    Returns dataset containing:
    image tensors
    label lists
    """
    # Use ImageFolder to create datasets. Make sure path has format: data/test/dog/image.jpg
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # Get class names
    class_names = train_data.classes

    # Loading the images into dataloader
    DataLoaderTrain = DataLoader(dataset = train_data,
                                batch_size = BATCH_SIZE,
                                shuffle = True, 
                                num_workers = NUM_WORKERS, 
                                pin_memory = True)

    DataLoaderTest = DataLoader(dataset = test_data, 
                                batch_size = BATCH_SIZE, 
                                shuffle = False, 
                                num_workers = NUM_WORKERS, 
                                pin_memory = True)


    return DataLoaderTrain, DataLoaderTest, class_names