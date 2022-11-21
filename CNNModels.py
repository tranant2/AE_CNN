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

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        
        # Encoder
        self.c2d1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=0)
        self.c2d2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(6272, latent_dim)
        
    def forward(self, xa):
        """
        Encode 33x33 image into latent_dim
        """
        xa = nn.functional.elu(self.c2d1(xa.view(-1,1,33,33)))
        xa = nn.functional.relu(self.c2d2(xa))
        xa = self.flatten(xa)
        xa = nn.functional.relu(self.linear1(xa))
        return xa
    
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        
        # Decoder
        self.linear2 = nn.Linear(latent_dim, 6272)
        self.ct2d1 = nn.ConvTranspose2d(128, 128, 3, stride=2, padding=0)
        self.ct2d2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=0)
        self.ct2d3 = nn.ConvTranspose2d(64, 1, 3, stride=1, padding=0)
        
    def forward(self, xb):
        """
        Given laten_dim, decode back 33x33 image
        """
        xb = nn.functional.relu(self.linear2(xb))
        xb = nn.functional.elu(self.ct2d1(xb.view(-1,128,7,7)))
        xb = nn.functional.relu(self.ct2d2(xb))
        xb = nn.functional.relu(self.ct2d3(xb))
        xb = torch.unsqueeze(torch.squeeze(xb), dim=3)  # make it shape (-1,33,33,1)
        return xb
    
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        latent_array = [32,16,16,16,16,32]
        # Encoders
        self.encoders = nn.ModuleList([Encoder(i) for i in latent_array])
        # Decoders
        self.decoders = nn.ModuleList([Decoder(i) for i in latent_array])
        # N latent dimention
        n = sum(latent_array)+6
        # NN transformers
        self.quadpair0 = nn.Linear(n, n)
        self.quadpair1 = nn.Linear(n, n)
        self.quadpair2 = nn.Linear(n, n)
        self.quadtriplet = nn.Linear(n, n)
        # NN 134->32-16-16-16-16-32
        self.linear = nn.ModuleList([nn.Linear(n, i) for i in latent_array])
        # Loss counter
#         self.linearLoss0_ = nn.Linear(n, n*4)
#         self.linearLoss1_ = nn.Linear(n, n*4)
#         self.linearLoss2_ = nn.Linear(n, n*4)
#         self.linearLoss3_ = nn.Linear(n, n*4)
#         self.linearLoss0 = nn.Linear(n*4, 1)
#         self.linearLoss1 = nn.Linear(n*4, 1)
#         self.linearLoss2 = nn.Linear(n*4, 1)
#         self.linearLoss3 = nn.Linear(n*4, 1)
        self.Loss0 = nn.Linear(n,n*4)
        self.Loss1 = nn.Linear(n*4,n*4)
        self.Loss2 = nn.Linear(n*4,1)
        
    def forward(self,voltages,
                distro0,distro1,distro2,distro3,distro4,
                output0,output1,output2,output3):
        x = [None]*6
        distros = [None]*5
        losses = [None]*4
        
        distros_0 = [None]*6
        for i in range(6):
            x[i] = nn.functional.relu(self.encoders[i](distro0[:,:,:,i]))  # Go throught each projections
            distros_0[i] = nn.functional.relu(self.decoders[i](x[i]))
        distros[0] = torch.cat(distros_0, axis=3)
        
        latentx0 = torch.cat(x, dim=1)
        latentx0 = torch.cat([latentx0,voltages], dim=1)
        
        latentx1 = nn.functional.relu(self.quadpair0(latentx0))
        distros_1 = [None]*6
        for i in range(6):
            distros_1[i] = nn.functional.relu(self.decoders[i](self.linear[i](latentx1))) # First reduce into 32 or 16, then decode into image.
        distros[1] = torch.cat(distros_1, axis=3)
        #losses[0] = nn.functional.relu(self.linearLoss0(latentx1))
        losses[0] = nn.functional.relu(self.Loss2(nn.functional.relu(self.Loss1(nn.functional.relu(self.Loss0(latentx1))))))
        
        latentx2 = nn.functional.relu(self.quadpair1(latentx1))
        distros_2 = [None]*6
        for i in range(6):
            distros_2[i] = nn.functional.relu(self.decoders[i](self.linear[i](latentx2))) # First reduce into 32 or 16, then decode into image.
        distros[2] = torch.cat(distros_2, axis=3)
        #losses[1] = nn.functional.relu(self.linearLoss1(latentx2) + losses[0])
        losses[1] = nn.functional.relu(self.Loss2(nn.functional.relu(self.Loss1(nn.functional.relu(self.Loss0(latentx2))))))
        
        latentx3 = nn.functional.relu(self.quadpair2(latentx2))
        distros_3 = [None]*6
        for i in range(6):
            distros_3[i] = nn.functional.relu(self.decoders[i](self.linear[i](latentx3))) # First reduce into 32 or 16, then decode into image.
        distros[3] = torch.cat(distros_3, axis=3)
        #losses[2] = nn.functional.relu(self.linearLoss2(latentx3) + losses[1])
        losses[2] = nn.functional.relu(self.Loss2(nn.functional.relu(self.Loss1(nn.functional.relu(self.Loss0(latentx3))))))
        
        latentx4 = nn.functional.relu(self.quadtriplet(latentx3))
        distros_4 = [None]*6
        for i in range(6):
            distros_4[i] = nn.functional.relu(self.decoders[i](self.linear[i](latentx4))) # First reduce into 32 or 16, then decode into image.
        distros[4] = torch.cat(distros_4, axis=3)
        #losses[3] = nn.functional.relu(self.linearLoss3(latentx4) + losses[2])
        losses[3] = nn.functional.relu(self.Loss2(nn.functional.relu(self.Loss1(nn.functional.relu(self.Loss0(latentx4))))))
        
        losses[0] = torch.squeeze(losses[0])
        losses[1] = torch.squeeze(losses[1])
        losses[2] = torch.squeeze(losses[2])
        losses[3] = torch.squeeze(losses[3])
        
        return distros, losses
