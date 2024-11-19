import numpy as np
import ipdb

import torch.utils.data as data

import nibabel as nib
import glob
import torch

class BUSLoader(data.Dataset):
    def __init__(self, option='train'):
        if option == 'train':
            self.data = glob.glob('path/to/train/*.png')
            self.transform = transform
        else:
            self.data = glob.glob('path/to/val/*.png')
            self.transform = None
        self.labels = []
        for path_ in self.data:
            if 'normal' in path_:
                lab = 0
            elif 'nodule' in path_:
                lab = 1
            else:
                print('error')  
            self.labels.append(lab)
        
    def __getitem__(self, index):
        img_path = self.data[index]
        img = nib.load(img_path).get_fdata()
        lab = self.labels[index]
        img = img/255.
        img = torch.Tensor(img)
        img = img.unsqueeze(0)
        return img, lab

    def __len__(self):
        return len(self.data)
