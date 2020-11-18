import os
import numpy as np
import torch
from torch.optim import Adam
from torchvision import models
from util.misc_functions import preprocess_image, recreate_image, save_image

class NeuralController():
    def __init__(self, model, selected_units, image_dim, device, w=1.,b=0.):
        # truncate model and set to evaluation mode
        self.model = model
        self.model.eval()
        self.w = torch.tensor(w,requires_grad=False).float().to(device)
        self.b = torch.tensor(b,requires_grad=False).float().to(device)
        self.dim = image_dim
        self.selected_units = selected_units
        self.conv_output = 0
        self.losses = []
        self.created_image = []
        self.device = device
        # Create the folder to export images if not exists
        if not os.path.exists('generated'):
            os.makedirs('generated')
            print('folder created')

    def visualize(self,niter=30,label='',obj = 'stretch'):
        random_image = np.uint8(np.random.uniform(60, 140, (self.dim[0], self.dim[1], self.dim[2])))
        self.processed_image = preprocess_image(random_image, self.device, False)
        # Define optimizer for the image
        optimizer = Adam([self.processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(niter):
            x = self.model.forward(self.processed_image)
            if torch.numel(self.w)==1:
                loss = -torch.mean(x[0, self.selected_units].squeeze())
            else:
                loss = -torch.mean(torch.matmul(self.w,x[0, self.selected_units].reshape(len(self.selected_units),-1)))-self.b
            loss.backward()
            optimizer.step()
        self.act = -loss.cpu().data.numpy()
        # Recreate image
        self.created_image = recreate_image(self.processed_image.cpu())
        # Save image
        im_path = 'generated\exp_' + label + '.jpg'
        save_image(self.created_image, im_path)