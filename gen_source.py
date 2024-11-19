import nibabel as nib
import numpy as np
import glob
import os
import cv2
import scipy.ndimage

def find_2D_box(label_img, relax=0):
    origin_size = label_img.shape
    idx = np.where(label_img!=0)
    x1, x2, y1, y2 = idx[0].min(), idx[0].max(), idx[1].min(), idx[1].max()
    x1, y1 = np.max([0,x1-relax]), np.max([0,y1-relax])
    x2, y2 = np.min([origin_size[0],x2+relax]), np.min([origin_size[1],y2+relax])
    return x1, x2, y1, y2

def find_3D_box(mask, relax=0):
    non_zero_indices = np.array(np.nonzero(mask))
    min_coords = np.min(non_zero_indices, axis=1)
    max_coords = np.max(non_zero_indices, axis=1)
    min_coords_relaxed = np.maximum(min_coords - relax, 0) 
    max_coords_relaxed = np.minimum(max_coords + relax, np.array(mask.shape) - 1) 
    x1, y1, z1 = min_coords_relaxed
    x2, y2, z2 = max_coords_relaxed
    return x1, x2, y1, y2, z1, z2

def process_image(img_path, mask_path, save_dir, direction='left'):
    img = cv2.imread(img_path, 0)
    mask = cv2.imread(mask_path, 0)
    x1, x2, y1, y2 = find_2D_box(mask)
    x_, y_ = x2 - x1, y2 - y1
    if direction == 'left':
        x1_, x2_, y1_, y2_ = x1, x2, max(0, y1 - y_), y1
        img_paste = img[x1_:x2_, y1_:y2_]
    elif direction == 'right':
        x1_, x2_, y1_, y2_ = x1, x2, y2, min(img.shape[1], y2 + y_)
        img_paste = img[x1_:x2_, y1_:y2_]
    elif direction == 'top':
        x1_, x2_, y1_, y2_ = max(0, x1 - x_), x1, y1, y2
        img_paste = img[x1_:x2_, y1_:y2_]
    elif direction == 'bottom':
        x1_, x2_, y1_, y2_ = x2, min(img.shape[0], x2 + x_), y1, y2
        img_paste = img[x1_:x2_, y1_:y2_]
    elif direction == 'left_right':
        y1_left, y2_left = max(0, y1 - y_), y1
        y1_right, y2_right = y2, min(img.shape[1], y2 + y_)
        left_patch = img[x1:x2, y1_left:y2_left]
        right_patch = img[x1:x2, y1_right:y2_right]
        left_patch_resized = cv2.resize(left_patch, (y2 - y1, x2 - x1), interpolation=cv2.INTER_NEAREST)
        right_patch_resized = cv2.resize(right_patch, (y2 - y1, x2 - x1), interpolation=cv2.INTER_NEAREST)
        img_paste = 0.5 * left_patch_resized + 0.5 * right_patch_resized
    elif direction == 'top_bottom':
        x1_top, x2_top = max(0, x1 - x_), x1
        x1_bottom, x2_bottom = x2, min(img.shape[0], x2 + x_)
        top_patch = img[x1_top:x2_top, y1:y2]
        bottom_patch = img[x1_bottom:x2_bottom, y1:y2]
        top_patch_resized = cv2.resize(top_patch, (y2 - y1, x2 - x1), interpolation=cv2.INTER_NEAREST)
        bottom_patch_resized = cv2.resize(bottom_patch, (y2 - y1, x2 - x1), interpolation=cv2.INTER_NEAREST)
        img_paste = 0.5 * top_patch_resized + 0.5 * bottom_patch_resized
    else:
        raise ValueError("Invalid direction specified.")
    img_paste = img[x1_:x2_, y1_:y2_]
    if img_paste.size != 0:
        img_paste = cv2.resize(img_paste, (y_, x_), interpolation=cv2.INTER_NEAREST)
    else:
        img_paste = np.ones((x_, y_)) * np.mean(img)
    img[x1:x2, y1:y2] = img_paste
    filename = os.path.splitext(os.path.basename(img_path))[0] + f'_{direction}.png'
    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, img)


def process_volume(volume_path, mask_path, save_dir, direction='left'):
    vol_sp_nii = nib.load(volume_path)
    mask_nii = nib.load(mask_path)
    
    vol_sp = vol_sp_nii.get_fdata()
    mask = mask_nii.get_fdata()
    
    x1, x2, y1, y2, z1, z2 = find_3D_box(mask)
    x_, y_, z_ = x2 - x1, y2 - y1, z2 - z1

    if direction == 'left':
        x1_, x2_, y1_, y2_, z1_, z2_ = x1, x2, max(0, y1 - y_), y1, z1, z2
        vol_paste = vol_sp[x1_:x2_, y1_:y2_, z1_:z2_]
    elif direction == 'right':
        x1_, x2_, y1_, y2_, z1_, z2_ = x1, x2, y2, min(vol_sp.shape[1], y2 + y_), z1, z2
        vol_paste = vol_sp[x1_:x2_, y1_:y2_, z1_:z2_]
    elif direction == 'top':
        x1_, x2_, y1_, y2_, z1_, z2_ = max(0, x1 - x_), x1, y1, y2, z1, z2
        vol_paste = vol_sp[x1_:x2_, y1_:y2_, z1_:z2_]
    elif direction == 'bottom':
        x1_, x2_, y1_, y2_, z1_, z2_ = x2, min(vol_sp.shape[0], x2 + x_), y1, y2, z1, z2
        vol_paste = vol_sp[x1_:x2_, y1_:y2_, z1_:z2_]
    elif direction == 'front':
        x1_, x2_, y1_, y2_, z1_, z2_ = x1, x2, y1, y2, max(0, z1 - z_), z1
        vol_paste = vol_sp[x1_:x2_, y1_:y2_, z1_:z2_]
    elif direction == 'back':
        x1_, x2_, y1_, y2_, z1_, z2_ = x1, x2, y1, y2, z2, min(vol_sp.shape[2], z2 + z_)
        vol_paste = vol_sp[x1_:x2_, y1_:y2_, z1_:z2_]
    elif direction == 'left_right':
        y1_left, y2_left = max(0, y1 - y_), y1
        y1_right, y2_right = y2, min(vol_sp.shape[1], y2 + y_)
        left_patch = vol_sp[x1:x2, y1_left:y2_left, z1:z2]
        right_patch = vol_sp[x1:x2, y1_right:y2_right, z1:z2]
        left_patch_resized = scipy.ndimage.zoom(left_patch, (
            x_ / left_patch.shape[0],
            y_ / left_patch.shape[1],
            z_ / left_patch.shape[2]), order=0)
        right_patch_resized = scipy.ndimage.zoom(right_patch, (
            x_ / right_patch.shape[0],
            y_ / right_patch.shape[1],
            z_ / right_patch.shape[2]), order=0)
        vol_paste = 0.5 * left_patch_resized + 0.5 * right_patch_resized
    elif direction == 'top_bottom':
        x1_top, x2_top = max(0, x1 - x_), x1
        x1_bottom, x2_bottom = x2, min(vol_sp.shape[0], x2 + x_)
        top_patch = vol_sp[x1_top:x2_top, y1:y2, z1:z2]
        bottom_patch = vol_sp[x1_bottom:x2_bottom, y1:y2, z1:z2]
        top_patch_resized = scipy.ndimage.zoom(top_patch, (
            x_ / top_patch.shape[0],
            y_ / top_patch.shape[1],
            z_ / top_patch.shape[2]), order=0)
        bottom_patch_resized = scipy.ndimage.zoom(bottom_patch, (
            x_ / bottom_patch.shape[0],
            y_ / bottom_patch.shape[1],
            z_ / bottom_patch.shape[2]), order=0)
        vol_paste = 0.5 * top_patch_resized + 0.5 * bottom_patch_resized
    elif direction == 'front_back':
        z1_front, z2_front = max(0, z1 - z_), z1
        z1_back, z2_back = z2, min(vol_sp.shape[2], z2 + z_)
        front_patch = vol_sp[x1:x2, y1:y2, z1_front:z2_front]
        back_patch = vol_sp[x1:x2, y1:y2, z1_back:z2_back]
        front_patch_resized = scipy.ndimage.zoom(front_patch, (
            x_ / front_patch.shape[0],
            y_ / front_patch.shape[1],
            z_ / front_patch.shape[2]), order=0)
        back_patch_resized = scipy.ndimage.zoom(back_patch, (
            x_ / back_patch.shape[0],
            y_ / back_patch.shape[1],
            z_ / back_patch.shape[2]), order=0)

        vol_paste = 0.5 * front_patch_resized + 0.5 * back_patch_resized

    vol_paste = vol_sp[x1_:x2_, y1_:y2_, z1_:z2_]

    if vol_paste.size != 0:
        vol_paste = scipy.ndimage.zoom(vol_paste, (x_, y_, z_), order=0)
    else:
        vol_paste = np.ones((x_, y_, z_)) * np.mean(vol_sp)

    vol_sp[x1:x2, y1:y2, z1:z2] = vol_paste

    vol_sp_img = nib.Nifti1Image(vol_sp, vol_sp_nii.affine)
    filename = os.path.splitext(os.path.basename(volume_path))[0] + f'_{direction}.nii'
    save_path = os.path.join(save_dir, filename)
    nib.save(vol_sp_img, save_path)

if __name__ == '__main__':
    import os
    import glob
    
    img_paths = glob.glob('path/to/img/*.png')  
    save_dir = 'path/to/save'
    os.makedirs(save_dir, exist_ok=True)
    directions = ['left', 'right', 'top', 'bottom', 'left_right', 'top_bottom']

    for img_path in img_paths:
        print(img_path)
        mask_path = 'path/to/mask/'
        for direction in directions:
            process_image(img_path, mask_path, save_dir, direction=direction)