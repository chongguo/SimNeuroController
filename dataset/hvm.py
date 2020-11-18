import torch
import matplotlib.image as mpimg
import os

class HVMDataset(torch.utils.data.Dataset):
    """HVM dataset from Dicarlo Lab"""

    def __init__(self,device,nsamp = 5000):
        self.img_dir = r'D:\chongguo\.brainio\image_dicarlo_hvm'
        img_list= [file for file in os.listdir(self.img_dir) if file.endswith('.png')]
        self.n = nsamp #len(img_list)
        self.dim = (3,256,256)
        self.data = torch.empty((self.n, 3, 256,256))
        for [index,imgname] in enumerate(img_list[:nsamp]):   
            self.data[index]=torch.from_numpy(mpimg.imread(self.img_dir+os.path.sep+imgname)).permute(2, 0, 1)
        self.data = self.data.to(device)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx]