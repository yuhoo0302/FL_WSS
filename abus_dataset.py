import numpy as np
import ipdb

import torch.utils.data as data

import nibabel as nib
import glob
import torch

def resize_and_pad(volume, target_shape):
    scale = min(target_shape[i] / volume.shape[i] for i in range(3))
    new_shape = tuple(int(volume.shape[i] * scale) for i in range(3))
    resized_volume = np.zeros(new_shape, dtype=volume.dtype)
    for z in range(new_shape[2]):
        for y in range(new_shape[1]):
            for x in range(new_shape[0]):
                src_x = min(int(x / scale), volume.shape[0] - 1)
                src_y = min(int(y / scale), volume.shape[1] - 1)
                src_z = min(int(z / scale), volume.shape[2] - 1)
                resized_volume[x, y, z] = volume[src_x, src_y, src_z]
    padding = [(target_shape[i] - resized_volume.shape[i]) // 2 for i in range(3)]
    pad_width = [(padding[i], target_shape[i] - resized_volume.shape[i] - padding[i]) for i in range(3)]
    padded_volume = np.pad(resized_volume, pad_width, mode='constant', constant_values=0)
    return padded_volume

class ABUSLoader(data.Dataset):
    def __init__(self, option='train'):
        if option == 'train':
            self.data = glob.glob('path/to/train/*.nii.gz')
            self.transform = None
        else:
            self.data = glob.glob('path/to/val/*.nii.gz')
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
        nii_path = self.data[index]
        vol = nib.load(nii_path).get_fdata()
        lab = self.labels[index]
        target_shape = (64, 64, 64)
        vol = resize_and_pad(vol, target_shape)
        vol = vol/255.
        vol = torch.Tensor(vol)
        vol = vol.unsqueeze(0)
        return vol, lab

    def __len__(self):
        return len(self.data)
