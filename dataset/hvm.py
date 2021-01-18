import torch
from matplotlib.pyplot import imread
from util.misc_functions import preprocess_image
import os
import random
from tqdm import tqdm

class HVMDataset(torch.utils.data.Dataset):
    """HVM dataset from Dicarlo Lab"""

    def __init__(self, device, n_samp = None, n_train = None, n_test = None, n_clamp = None, shuffle=True, subset = 'all'):
        self.img_dir = r'D:\chongguo\.brainio\image_dicarlo_hvm'
        img_list= [file for file in os.listdir(self.img_dir) if file.endswith('.png')]
        if n_samp is None:
            self.n = len(img_list)
        else:
            self.n = n_samp
        self.dim = imread(self.img_dir+os.path.sep+img_list[0]).shape
        self.shuffled_idx = list(range(self.n));
        self.n_train = n_train
        self.n_test = n_test
        self.n_clamp = n_clamp
        if shuffle:
            random.seed(0)
            self.shuffled_idx = random.sample(self.shuffled_idx,self.n)
        # read images
        print('loading and preprocessing hvm')
        if subset == 'clamp':
            self.n = n_clamp
            self.data = torch.empty((self.n, self.dim[2], self.dim[0], self.dim[1])).to(device)
            for [index,imgname] in enumerate(tqdm(img_list[:n_samp])):   
                if (self.shuffled_idx[index]>=n_train) & (self.shuffled_idx[index]<(n_train+n_clamp)):
                    im = imread(self.img_dir+os.path.sep+imgname)
                    self.data[self.shuffled_idx[index]-n_train]=preprocess_image(im, dataset = 'hvm', device = device, asvar=False,unsqueeze=False)
        else:
            self.data = torch.empty((self.n, self.dim[2], self.dim[0], self.dim[1])).to(device)
            for [index,imgname] in enumerate(tqdm(img_list[:n_samp])):   
                im = imread(self.img_dir+os.path.sep+imgname)
                self.data[self.shuffled_idx[index]]=preprocess_image(im, dataset = 'hvm', device = device, asvar=False,unsqueeze=False)
            
    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx]