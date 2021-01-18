import os
import numpy as np
import torch
from torch.optim import Adam, SGD
from torchvision import models
from util.misc_functions import preprocess_image, recreate_image, save_image, indexfun
import warnings 

class NeuralController():
    def __init__(self, model, indexer, img_dim, device):
        # this controller takes in pre-trimmed target or surrogate models
        # only clamp/control units from the last layer
        self.model = model
        self.img_dim = img_dim
        self.indexer = indexer
        self.device = device
        # initialize outputs
        self.loss = []
        self.act = []
        self.img = []
        # set model to evaluation mode
        self.model.to(device)
        self.model.eval()

    def scalar(self, selected_unit, target_act, niter=20,label='',obj = 'clamp'):
        # scalar control of a selected neuron
        # two options for obj:
        # 1. strech -  drive up unit activation as much as possible
        # 2. clamp - clamp single unit activation to a specified target
        self.img = torch.FloatTensor(1, self.dim[0], self.dim[1], self.dim[2],requires_grad = True).uniform_(60, 140).to(self.device)
        # Define optimizer for the image
        optimizer = Adam([self.processed_image], lr=0.01, weight_decay=1e-5)
        for i in range(niter):
            x = self.model.forward(self.processed_image)
            loss = -torch.mean(x[0, self.selected_unit].squeeze())
            loss.backward()
            optimizer.step()
        self.loss = -loss.cpu().data.numpy()
        
    def vector(self, selected_units, target, obj = 'clamp', niter=20, tol=1e3):
        # vector control of neural populations
        # two options for obj:
        # 1. ohp - one-hot-population control with CE loss on a subset of neurons
        # 2. clamp - clamp population activtion at a sepcified vectored target 
        self.img = torch.FloatTensor(1, self.dim[0], self.dim[1], self.dim[2], requires_grad = True).uniform_(60, 140).to(self.device)
        # Define optimizer for the image
        optimizer = Adam(self.img,lr=0.01,weight_decay=1e-2)
        if obj=='ohp':
            vecloss = nn.CrossEntropyLoss()
        elif obj=='clamp':
            vecloss = nn.MSELoss()
        else:
            warnings.warn('Option ('+ obj +') is not supported.')
        for i in range(niter):
            x = self.model.forward(self.img)
            loss = vecloss(torch.x[0,self.selected_units],target)
            loss.backward()
            optimizer.step()
        self.loss = loss.cpu().data.numpy()