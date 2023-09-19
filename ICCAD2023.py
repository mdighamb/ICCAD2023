import torch
import math
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
from sklearn.model_selection import train_test_split
import scipy
import matplotlib.pyplot as plt
import os
import pickle

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

burgersdata = 'burgers_data_R10.mat'
nsdata = 'NavierStokes_V1e-5_N1200_T20.mat'

inputdata = input('Select dataset (Darcy, Burgers, or NS): ')
inputmodel = input('Select model (fno or cnn): ')

def noisify(tensor,sigma):
    noise_tensor = torch.randn(32, 1, 16, 16) * sigma
    return tensor + noise_tensor

def read_data():
    if inputdata == 'Burgers':
        data = scipy.io.loadmat(burgersdata)
    elif inputdata == 'NS':
        data = scipy.io.loadmat(nsdata)
    else:
        raise AssertionError('Invalid Input Data') 
    return data['a'], data['u']

def load_darcy():
    train_loader, test_loaders, output_encoder = load_darcy_flow_small(
            n_train=1000, batch_size=32,
            test_resolutions=[16], n_tests=[100, 50],
            test_batch_sizes=[32, 32],
    )
    return train_loader, test_loaders, output_encoder



def tensorize(trainratio,a,u):
    
    if inputdata =='Burgers':
        shape = 2048
        ulayers = 1
    elif inputdata == 'NS':
        shape = 600
        ulayers = 20

    # Reshape the original array into a shape that can be divided into batches
    a_r = a.reshape(shape, 32, 1, 16, 16)
    u_r = u.reshape(shape, 32, ulayers, 16, 16)
    
    # Split the reshaped array into batches along the first dimension (axis 0)
    a_batches = np.split(a_r, shape, axis=0)
    u_batches = np.split(u_r, shape, axis=0)
    
    
    a_stack = np.vstack(a_batches)
    u_stack = np.vstack(u_batches)
    
    a_final = [item.reshape(32, 1, 16, 16) for item in a_stack]
    u_final = [item.reshape(32, ulayers, 16, 16) for item in u_stack]
    
    a_final = a_final[:]#[:Burgers]
    u_final = u_final[:]#[:Burgers]
    
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


if inputdata != 'Darcy':
    a, u = read_data()
    train_x, train_y, test_x, test_y = tensorize(trainratio,a,u)
    train = {'x': train_x, 'y': train_y}
    test = {'x': test_x, 'y': test_y}

else:
    train_loader, test_loaders, output_encoder = load_darcy()



def initialize_model():
    if inputmodel == 'fno':
        if inputdata == 'Darcy':
            model = TFNO(n_modes=(16, 16), hidden_channels=32, projection_channels=64, factorization='tucker', rank=0.42)
        else:
            if inputdata == 'NS':
                out = 1
            elif inputdata == 'Burgers':
                out = 20
            model = TFNO(n_modes=(16, 16), hidden_channels=64,
                            in_channels=1,
                            out_channels=out,
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
    
    return model
model = initialize_model()
model = model.to(device)



cutoff = 40#math.floor(trainratio * 2048)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(),
                                lr=8e-3,
                                weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

h1loss = H1Loss(d=2)
l2loss = LpLoss(d=2, p=2)






def trainer(num_epochs, sigma):
    if inputdata != 'Darcy':
        # Define the number of training epochs
        # Training loop
        newinputs = []
        for i in range(cutoff):
            newinputs.append(noisify(train_x[i], sigma))
        
        
        for epoch in range(num_epochs):
            # Set the model to training mode
            model.train()
            # Iterate over the training dataset
            for i in range(cutoff):
    
                # Clear the gradients
                optimizer.zero_grad()
    
                # Forward pass
                inputs = newinputs[i]
                pred = model(inputs)
                true = train_y[i]
                # Compute the loss
               
                loss_value = h1loss(pred, true)
    
                # Backward pass and optimization
                loss_value.sum().backward()
                #loss_value.mean().backward(),
                optimizer.step()
            print('Epoch: ' + str(epoch))
    
        # Set the model to evaluation mode
        model.eval()
    
        return model, pred, true

    else:
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



'''
if inputdata == 'Darcy':
    model20, pred, true = darcytrainer(20)
else:
    model20, pred, true = darcytrainer(20)



filename = 'trained_' + str(inputmodel) + '_' + str(inputdata) + '.pkl'

if not os.path.exists(filename):
    with open(filename, 'wb') as file:
        pickle.dump(model20, file)
'''


    
    
    
def tester():

    test_loss = 0
    loss2 = 0
    with torch.no_grad():
        for i in range(len(test_x)):
            
            inputs = test['x'][i]

            pred = model(inputs)
            ground_truth = test['y'][i]

            loss_tensor = h1loss(pred, ground_truth)
            test_loss += loss_tensor.sum().detach().numpy().item()

            loss2 += torch.mean(l2loss(pred,ground_truth)).item()
            
    test_loss = test_loss / len(test_x)
    loss2 = loss2 / len(test_x)
    return test_loss, loss2 # H1loss, L2loss
    


for i in range(21):

    noise = i * 0.05 
    
    filename = 'trained_' + str(inputmodel) + '_' + str(inputdata) + '_' + str(i) + '.pkl'
    
    if not os.path.exists(filename):
        model = initialize_model()
        model = trainer(20, noise)[0]
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
    else:
        with open(filename, 'rb') as file:
                model = pickle.load(file)
    
    
    
    result = tester()
    print(noise, result[0], result[1])
            
            
            
            
            
            
            
            