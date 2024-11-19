from skimage.segmentation import slic, mark_boundaries, find_boundaries
import cv2
import numpy as np
import torch
import nibabel as nib
import torch.nn.functional as F

def find_3D_box(mask, relax=0):
    non_zero_indices = np.array(np.nonzero(mask))
    min_coords = np.min(non_zero_indices, axis=1)
    max_coords = np.max(non_zero_indices, axis=1)
    min_coords_relaxed = np.maximum(min_coords - relax, 0) 
    max_coords_relaxed = np.minimum(max_coords + relax, np.array(mask.shape) - 1) 
    x1, y1, z1 = min_coords_relaxed
    x2, y2, z2 = max_coords_relaxed
    return x1, x2, y1, y2, z1, z2


def supervoxel(path, mask_path, save_path, n_segments=10000, relax=5, is_save=False):
    vol = nib.load(path)
    seg_mask = nib.load(mask_path)
    seg_data = seg_mask.get_fdata()
    x1, x2, y1, y2, z1, z2 = find_3D_box(seg_data,relax=relax)
    vol_crop = vol.get_fdata()[x1:x2, y1:y2, z1:z2]
    vol_resize = F.interpolate(torch.Tensor([[vol_crop]]), size=(100,100,100), mode='trilinear', align_corners=False) 
    vol_resize = vol_resize[0][0].numpy()
    vol_sp = slic(vol_resize.astype('uint8'), n_segments=n_segments, compactness=.1,start_label=1,multichannel=False)
    if if_save:
        vol_sp_bound = F.interpolate(torch.Tensor([[vol_sp]]), size=shape_shape, mode='nearest')
        vol_sp_bound = vol_sp_bound.numpy()[0][0]
        vol_sp_bound = find_boundaries(vol_sp_bound,mode='inner')
        vol_sp_bound = nib.Nifti1Image(vol_sp_bound.astype(np.int32), affine=vol.affine)
        nib.save(vol_sp_bound,filename=save_path)

if __name__ == '__main__':
    img_path = 'path/to/img'
    mask_path = 'path/to/mask'  
    save_path = 'path/to/save'
    n_segments = 100
    relax = 0
    supervoxel(img_path, mask_path, save_path, n_segments=n_segments, relax=relax, is_save=True)