import torch
import math
import matplotlib.pyplot as plt
import sys
from neuralop.models import TFNO, TFNO2d
from neuralop.datasets import load_darcy_flow_small
from neuralop import LpLoss, H1Loss
from neuralop.datasets import load_navier_stokes_pt
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import tensorflow as tf

import numpy as np
from sklearn.model_selection import train_test_split
import scipy
import matplotlib.pyplot as plt
import os
import pickle

device = 'cpu'

v1000 = 'burgers_v1000_t200_r1024_N2048.mat'
v100 = 'burgers_v100_t100_r1024_N2048.mat'
r10 = 'burgers_data_R10.mat'


inputmodel = input('Select model (fno or cnn): ')

def noisify(tensor,sigma):
    noise_tensor = torch.randn(32, 1, 16, 16) * sigma
    return tensor + noise_tensor

def read_burgers():
    data = scipy.io.loadmat(r10)
    #print(data.keys())
    #print(data['a'].shape)
    #print(data['a_smooth'].shape)
    #print(data['a_smooth_x'].shape)
    #print(data['a_x'].shape)
    #print(data['u'].shape)
    return data['a'], data['u']



def tensorize(trainratio,a,u):

    # Reshape the original array into a shape that can be divided into batches
    a_r = a.reshape(2048, 32, 1, 16, 16)
    u_r = u.reshape(2048, 32, 1, 16, 16)
    
    # Split the reshaped array into batches along the first dimension (axis 0)
    a_batches = np.split(a_r, 2048, axis=0)
    u_batches = np.split(u_r, 2048, axis=0)
    
    
    a_stack = np.vstack(a_batches)
    u_stack = np.vstack(u_batches)
    
    a_final = [item.reshape(32, 1, 16, 16) for item in a_stack]
    u_final = [item.reshape(32, 1, 16, 16) for item in u_stack]
    
    a_final = a_final[:50]
    u_final = u_final[:50]
    
    cutoff = math.floor(trainratio * len(u_final))
    
    train_x = a_final[:cutoff]
    train_y = u_final[:cutoff]
    
    test_x = a_final[cutoff:]
    test_y = u_final[cutoff:]
    
    
    train_x = [torch.from_numpy(item).to(dtype=torch.float32) for item in train_x]
    train_y = [torch.from_numpy(item).to(dtype=torch.float32) for item in train_y]
    
    test_x = [torch.from_numpy(item).to(dtype=torch.float32) for item in test_x]
    test_y = [torch.from_numpy(item).to(dtype=torch.float32) for item in test_y]
    
    
    
    return train_x, train_y, test_x, test_y

trainratio = 0.8



a, u = read_burgers()

train_x, train_y, test_x, test_y = tensorize(trainratio,a,u)

train = {'x': train_x, 'y': train_y}
test = {'x': test_x, 'y': test_y}


if inputmodel == 'fno':
    model = TFNO(n_modes=(16, 16), hidden_channels=64,
                    in_channels=1,
                    out_channels=1,
                    factorization='tucker',
                    implementation='factorized',
                    rank=0.05)
elif inputmodel == 'cnn':
    model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
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



cutoff = math.floor(trainratio * 2048)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(),
                                lr=8e-3,
                                weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

h1loss = H1Loss(d=2)
l2loss = LpLoss(d=2, p=2)







def trainer(num_epochs, sigma):

   
    # Define the number of training epochs
    # Training loop
    for epoch in range(num_epochs):
        # Set the model to training mode
        model.train()
        # Iterate over the training dataset
        for i in range(cutoff):

            # Clear the gradients
            optimizer.zero_grad()

            # Forward pass
            inputs = noisify(train_x[i], sigma)
            pred = model(inputs)
            true = train_y[i]
            # Compute the loss
           
            loss_value = h1loss(pred, true)

            # Backward pass and optimization
            loss_value.sum().backward()
            #loss_value.mean().backward(),
            optimizer.step()

        
           

    # Set the model to evaluation mode
    model.eval()

    return model, pred, true


model20, pred, true = trainer(20)



filename = 'trained_fno.pkl'

if not os.path.exists(filename):
    with open(filename, 'wb') as file:
        pickle.dump(model20, file)
  


    
    
    
def tester():

    test_loss = 0
    loss2 = 0
    with torch.no_grad():
        for i in range(len(test_x)):
            
            inputs = test['x'][i]

            pred = model20(inputs)
            ground_truth = test['y'][i]

            loss_tensor = h1loss(pred, ground_truth)
            test_loss += loss_tensor.sum().detach().numpy().item()
            loss2 += l2loss(pred,ground_truth).item()
            
    test_loss = test_loss / len(test_x)
    loss2 = loss2 / len(test_x)
    return test_loss, loss2 # H1loss, L2loss
    


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
            
            
            
            
            
            
            
            