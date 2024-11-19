from skimage.segmentation import slic, mark_boundaries
import cv2
import numpy as np


def find_2D_box(label_img, relax=0):
    origin_size = label_img.shape
    idx = np.where(label_img!=0)
    x1, x2, y1, y2 = idx[0].min(), idx[0].max(), idx[1].min(), idx[1].max()
    x1, y1 = np.max([0,x1-relax]), np.max([0,y1-relax])
    x2, y2 = np.min([origin_size[0],x2+relax]), np.min([origin_size[1],y2+relax])
    return x1, x2, y1, y2



def superpixel(path, mask_path, save_path, n_segments=10000, relax=5):
    img = cv2.imread(path, 0)
    mask = cv2.imread(mask_path, 0)
    mask[mask>127] = 255
    mask[mask<=127] = 0
    x1, x2, y1, y2 = find_2D_box(mask, relax=relax)
    img_crop = img[x1:x2, y1:y2]
    img_crop = cv2.resize(img_crop, (100,100), interpolation = cv2.INTER_LINEAR)
    img_sp = slic(img_crop.astype('float'), n_segments=n_segments,max_iter=20, compactness=10,start_label=1)
    img_sp_boundary=mark_boundaries(img_crop,img_sp,mode='subpixel')
    cv2.imwrite(save_path + 'sp_b.png'.format(bg_type), img_sp_boundary*255)
    

if __name__ == '__main__':
    img_path = 'path/to/img'
    mask_path = 'path/to/mask'  
    save_path = 'path/to/save'
    n_segments = 100
    relax = 0
    superpixel(img_path, mask_path, save_path, n_segments=n_segments, relax=relax)