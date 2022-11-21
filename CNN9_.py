import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Neural networks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import torchvision

import h5py
import os

from H5Dataset import H5Dataset
from CNNModels import *
from CNNTrain import *

np.random.seed(0) # set seed for reproduceability
test_number = 1
epoch0 = 20
epoch1 = 10
epoch2 = 5

# Make a function to get data from a list of files
hdf5_dir = "/mnt/gs18/scratch/users/tranant2/Distro_LessVariation2"
files = os.listdir(hdf5_dir)
hdf5_files1 = [(hdf5_dir + '/'+i) for i in files ]

for file in hdf5_files1:
    print(file)
    
dataset = H5Dataset(hdf5_files1[0:], transform=False)  # Change these as well, #110 is 186 GB

ntrain = int(dataset.__len__()*.8)
ntest = dataset.__len__() - ntrain

train_set, test_set = torch.utils.data.random_split(dataset,[ntrain,ntest])
train_loader = DataLoader(dataset=train_set, batch_size = 1024, shuffle=True)  # And also change these some times
test_loader = DataLoader(dataset=test_set, batch_size = 1024, shuffle=True)

print(f"Is GPU avaliable: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Got the: {torch.cuda.get_device_name(0)}")
    print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory/(1028**3)} GB")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Computation will be happening on the {device}")
#-------------------------------------------------------------------Training 2--------------------------------------------------------
print("On to state 2 of training. Training the loss function")
model = VAE().to(device)
model.load_state_dict(torch.load( f"/mnt/ufs18/home-032/tranant2/Desktop/MachineLearning/TRACK/CNN9/model_{test_number}.pth"), strict=False)

# Code to get params
params = model.state_dict()
keys = list(params.keys())
for key in keys[:-6]:
    print(key)
    
# Code add back gradient
for name, param in model.named_parameters():
    if not param.requires_grad:
        param.requires_grad = True
        
# Code to remove remove gradient from auto-encoder
for name, param in model.named_parameters():
    if name in keys[:-6] and param.requires_grad:
        param.requires_grad = False
        
optimizer2 = optim.Adam(filter(lambda p: p.requires_grad, model.to(device).parameters()), lr=.01,
                       betas=(.950,.970),
                       eps=1e-9,
                       weight_decay = 0.000,
                       amsgrad=False,)

n_epochs = epoch1
train_losses, counter = train_model2(model, n_epochs, train_loader, optimizer2, device=device)

fig, ax = plt.subplots(1,1,figsize=(8, 8))  # a figure with a single Axes
plt.plot(train_losses)
fig.savefig(f"/mnt/home/tranant2/Desktop/MachineLearning/TRACK/CNN9/loss_{test_number}_2.png", dpi = 100)

examples = enumerate(test_loader)
batch_idx, (test_data, test_targets) = next(examples)

voltages = test_data[0].to(device)
distro_0 = test_data[1].to(device)
distro_1 = test_data[2].to(device)
distro_2 = test_data[3].to(device)
distro_3 = test_data[4].to(device)
distro_4 = test_data[5].to(device)
output_0 = test_targets[0].to(device)
output_1 = test_targets[1].to(device)
output_2 = test_targets[2].to(device)
output_3 = test_targets[3].to(device)
with torch.no_grad():
    distro, losses = model(voltages=voltages,
                           distro0=distro_0,
                           distro1=distro_1,
                           distro2=distro_2,
                           distro3=distro_3,
                           distro4=distro_4,
                           output0=output_0,
                           output1=output_1,
                           output2=output_2,
                           output3=output_3)
loss0 =nn.L1Loss(reduction='mean')(output_0, losses[0])
loss1 =nn.L1Loss(reduction='mean')(output_1, losses[1])
loss2 =nn.L1Loss(reduction='mean')(output_2, losses[2])
loss3 =nn.L1Loss(reduction='mean')(output_3, losses[3])
print(f"loss0:{loss0},loss1:{loss1}, loss2:{loss2}, loss3:{loss3},\n distro0:{0}, distro1:{0}, distro2:{0}, distro3:{0}, distro4:{0}")
    
fig = plt.figure(figsize=(12,12))
nimg = 5
for j in range(5):
    for i in range(6):
        plt.subplot(5,6,i+1+j*6)
        plt.tight_layout()
        plt.imshow(np.array(distro[j].cpu())[nimg,:,:,i], cmap='viridis', interpolation='none')
        plt.title(f"n:{i+1+j*6}")
        plt.xticks([])
        plt.yticks([])
fig.savefig(f"/mnt/home/tranant2/Desktop/MachineLearning/TRACK/CNN9/ReconstructedPhaseSpace_{test_number}_2.png", dpi = 100)

fig = plt.figure(figsize=(12,12))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.tight_layout()
    plt.scatter(losses[i].cpu(),test_targets[i])
    plt.xlabel('predicted')
    plt.ylabel('true')
    plt.xlim(0,10000)
    plt.ylim(0,10000)
fig.savefig(f"/mnt/home/tranant2/Desktop/MachineLearning/TRACK/CNN9/Losses_{test_number}_2.png", dpi = 100)

torch.save(model.state_dict(), f"/mnt/ufs18/home-032/tranant2/Desktop/MachineLearning/TRACK/CNN9/model__{test_number}.pth")

#-------------------------------------------------------------------Training 3--------------------------------------------------------
print("Finally, on to training 3: To fine tune things. Makue sure no overtraining")

model.load_state_dict(torch.load( f"/mnt/ufs18/home-032/tranant2/Desktop/MachineLearning/TRACK/CNN9/model__{test_number}.pth"), strict=False)

# Code add back gradient
for name, param in model.named_parameters():
    if not param.requires_grad:
        param.requires_grad = True
        
optimizer3 = optim.Adam(filter(lambda p: p.requires_grad, model.to(device).parameters()), lr=0.001,
                       betas=(.950,.970),
                       eps=1e-9,
                       weight_decay = 0.000,
                       amsgrad=False,)

n_epochs = epoch2
train_losses, counter = train_model3(model, n_epochs, train_loader, optimizer3, device=device)

fig, ax = plt.subplots(1,1,figsize=(8, 8))  # a figure with a single Axes
plt.plot(train_losses)
fig.savefig(f"/mnt/home/tranant2/Desktop/MachineLearning/TRACK/CNN9/loss_{test_number}_3.png", dpi = 100)

examples = enumerate(test_loader)
batch_idx, (test_data, test_targets) = next(examples)

voltages = test_data[0].to(device)
distro_0 = test_data[1].to(device)
distro_1 = test_data[2].to(device)
distro_2 = test_data[3].to(device)
distro_3 = test_data[4].to(device)
distro_4 = test_data[5].to(device)
output_0 = test_targets[0].to(device)
output_1 = test_targets[1].to(device)
output_2 = test_targets[2].to(device)
output_3 = test_targets[3].to(device)
with torch.no_grad():
    distro, losses = model(voltages=voltages,
                           distro0=distro_0,
                           distro1=distro_1,
                           distro2=distro_2,
                           distro3=distro_3,
                           distro4=distro_4,
                           output0=output_0,
                           output1=output_1,
                           output2=output_2,
                           output3=output_3)
loss0 = nn.L1Loss(reduction='mean')(output_0, losses[0])
loss1 = nn.L1Loss(reduction='mean')(output_1, losses[1])
loss2 = nn.L1Loss(reduction='mean')(output_2, losses[2])
loss3 = nn.L1Loss(reduction='mean')(output_3, losses[3])
loss4 = nn.MSELoss()(distro_0, distro[0])
loss5 = nn.MSELoss()(distro_1, distro[1])
loss6 = nn.MSELoss()(distro_2, distro[2])
loss7 = nn.MSELoss()(distro_3, distro[3])
loss8 = nn.MSELoss()(distro_4, distro[4])
print(f"loss0:{loss0},loss1:{loss1}, loss2:{loss2}, loss3:{loss3},\n distro0:{loss4}, distro1:{loss5}, distro2:{loss6}, distro3:{loss7}, distro4:{loss8}")

fig = plt.figure(figsize=(12,12))
nimg = 5
for j in range(5):
    for i in range(6):
        plt.subplot(5,6,i+1+j*6)
        plt.tight_layout()
        plt.imshow(np.array(distro[j].cpu())[nimg,:,:,i], cmap='viridis', interpolation='none')
        plt.title(f"n:{i+1+j*6}")
        plt.xticks([])
        plt.yticks([])
fig.savefig(f"/mnt/home/tranant2/Desktop/MachineLearning/TRACK/CNN9/ReconstructedPhaseSpace_{test_number}_3.png", dpi = 100)

fig = plt.figure(figsize=(12,12))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.tight_layout()
    plt.scatter(losses[i].cpu(),test_targets[i])
    plt.xlabel('predicted')
    plt.ylabel('true')
    plt.xlim(0,10000)
    plt.ylim(0,10000)
fig.savefig(f"/mnt/home/tranant2/Desktop/MachineLearning/TRACK/CNN9/Losses_{test_number}_3.png", dpi = 100)

torch.save(model.state_dict(), f"/mnt/ufs18/home-032/tranant2/Desktop/MachineLearning/TRACK/CNN9/model___{test_number}.pth")