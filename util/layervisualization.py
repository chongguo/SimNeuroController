"""
Created on Sat Nov 18 23:12:08 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import numpy as np

import torch
from torch.optim import Adam
from torchvision import models
from util.misc_functions import preprocess_image, recreate_image, save_image


class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter, image_dim, device):
        self.model = model
        self.model.eval()
        self.device = device
        self.dim = image_dim
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        self.losses = []
        self.created_image = []
        # Create the folder to export images if not exists
        if not os.path.exists('generated'):
            os.makedirs('generated')
            print('folder created')

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]
        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self,niter=100):
        # Hook the selected layer
        self.hook_layer()
        # Generate a random image
        random_image = np.uint8(np.random.uniform(100, 155, (self.dim[0], self.dim[1], self.dim[2])))
        # Process image and return variable
        processed_image = preprocess_image(random_image,self.device, False)
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=.05,weight_decay=1e-9)
        for i in range(1, niter+1):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
                x = layer(x)
                # Only need to forward until the selected layer is reached
                if index == self.selected_layer:
                    # (forward hook function triggered)
                    break
            self.conv_output = x[:, self.selected_filter]
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            self.losses.append(-loss.cpu().data.numpy())
            if i%50 == 0:
                print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.cpu().data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(processed_image.cpu())
            # Save image
            im_path = 'generated\layer_vis_l' + str(self.selected_layer) + \
                '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
            save_image(self.created_image, im_path)
        return self.created_image, self.losses