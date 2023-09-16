#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 10:04:54 2023

@author: mohitdighamber
"""

import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import TFNO
from neuralop.datasets import load_darcy_flow_small
from neuralop import LpLoss, H1Loss

import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import tensorflow as tf

import numpy as np
import pickle
import os


device = 'cpu'


class NoisyDataLoader(DataLoader):
    def __init__(self, dataloader, mean=0, std=1):
        super().__init__(dataset=dataloader.dataset, batch_size=dataloader.batch_size)
        self.dataloader = dataloader
        self.mean = mean
        self.std = std

    def __iter__(self):
        # Get the iterator from the original DataLoader
        dataloader_iter = iter(self.dataloader)

        # Iterate over the batches
        for batch in dataloader_iter:
            if isinstance(batch, dict) and 'x' in batch:
                # If the batch is a dictionary, add Gaussian noise to each value
                
                noisy_batch = {'x': batch['x'] + torch.randn_like(batch['x']) * self.std + self.mean, 'y': batch.get('y', None)}
            else:
                # If the batch is a tensor, add Gaussian noise to the tensor
                noisy_batch = batch# + torch.randn_like(batch) * self.std + self.mean
            yield noisy_batch

inputmodel = input('Enter model (fno or cnn): ')


train_loader, test_loaders, output_encoder = load_darcy_flow_small(
        n_train=1000, batch_size=32,
        test_resolutions=[16], n_tests=[100, 50],
        test_batch_sizes=[32, 32],
)

if inputmodel == 'fno':
    model = TFNO(n_modes=(16, 16), hidden_channels=32, projection_channels=64, factorization='tucker', rank=0.42)
elif inputmodel == 'cnn':
    
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16,16)
    )
    
else:
    raise AssertionError('Invalid Model Input')

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(),
                                lr=8e-3,
                                weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)




h1loss = H1Loss(d=2)
l2loss = LpLoss(d=2, p=2)

def blah():
    
    noisy_dataloader = NoisyDataLoader(train_loader, mean=0, std=1000)
    
    
    
    for batch_idx, samples in enumerate(noisy_dataloader):
        
        return samples
    


def trainer(num_epochs, sigma):

    noisy_dataloader = NoisyDataLoader(train_loader, mean=0, std=sigma)

    # Define the number of training epochs
    # Training loop
    for epoch in range(num_epochs):
        # Set the model to training mode
        model.train()
        # Iterate over the training dataset
        for batch_idx, samples in enumerate(noisy_dataloader):

            # Clear the gradients
            optimizer.zero_grad()

            # Forward pass
            pred = model(samples['x'])
            true = samples['y']
            # Compute the loss
           
            loss_value = h1loss(pred, true)

            # Backward pass and optimization
            loss_value.sum().backward()
            #loss_value.mean().backward(),
            optimizer.step()
           
            if batch_idx == 30:
                break
    # Set the model to evaluation mode
    model.eval()

    return model, pred, true


model20, pred, true = trainer(20)


filename = 'trained_' + str(inputmodel) + '.pkl'

if not os.path.exists(filename):
    with open(filename, 'wb') as file:
        pickle.dump(model20, file)
  


with open(filename, 'rb') as file:
        model20 = pickle.load(file)
        
        

def tester():
    test_loss = 0
    l2_loss = 0
    with torch.no_grad():
        for batch_idx, samples in enumerate(test_loaders[16]):
            datalen = batch_idx
            inputs = samples['x']

            pred = model(inputs)
            ground_truth  = samples['y']
            
            loss_tensor = h1loss(pred, ground_truth)
            test_loss += loss_tensor.sum().detach().numpy().item()
            l2_loss += l2loss(pred,ground_truth).item()
    
    test_loss = test_loss / datalen
    l2_loss = l2_loss / datalen
    return test_loss, l2_loss # H1loss, L2loss



for i in range(21):

    noise = i * 0.05 
    if inputmodel == 'fno':
        model = TFNO(n_modes=(16, 16), hidden_channels=32, projection_channels=64, factorization='tucker', rank=0.42)
    elif inputmodel == 'cnn':
        
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16,16)
        )
        
    else:
        raise AssertionError('Invalid Model Input')            
    model = trainer(20, noise)[0]
    result = tester()
    print(noise, result[0], result[1])
            
            
            
            
            
            
            
            
            
            
            
            
            

