import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import torch 
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import csv
# from data_loaders import test_data_loader, train_data_loader

# Create a dataset using PyTorch    
class CNNDataset(Dataset):
    def __init__(self, data_files):
        self.data_files = pd.read_csv(data_files)
        
    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data = pd.read_csv(self.data_files.loc[idx, 1:])
        X = torch.tensor(data.values)
        label = pd.read_csv(self.data_files.loc[idx]["label"])
        return data, label
    
# Create Dataloader 

# Create a CNN model
class CNN2D(torch.nn.Module):
    def __init__(self):
        self.conv_layer_1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2)
        self.conv_layer_2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=2)
        self.conv_layer_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2) 
        self.fc1 = nn.Linear()
        self.dropout = torch.nn.Dropout(0.2)
        self.pool_layer = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool_layer(F.relu(self.conv_layer_1(x)))
        x = self.pool_layer(F.relu(self.conv_layer_2(x)))
        x = self.pool_layer(F.relu(self.conv_layer_3(x)))
        x = self.fc1()
        x = self.fc2()
        return x
        
# For Pytorch Lightning, create a module that wraps the CNN model
class CNN2DModule(pl.LightningModule):
    def __init__(self, loss, weight_decay, learning_rate, scheduler):
        super(CNN2DModule, self).__init__()
# Create a trainer

def main():
    train_data = pathlib.Path(__file__).resolve().parent.joinpath(r'Data/train.csv')
    test_data = pd.read_csv(train_data)
    test_data.loc[1,:].to_numpy().reshape(28,28,1)
