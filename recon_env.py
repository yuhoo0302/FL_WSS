import numpy as np


def recon_image(img_sp):
    mid_x, mid_y = int(img_sp.shape[0]/2),int(img_sp.shape[1]/2)
    index_io = []
    for i in range(0,max(mid_x, mid_y)):
        mid_x_m, mid_y_m= max(mid_x - i,0), max(mid_y - i,0)
        mid_x_p, mid_y_p = min(mid_x + i,img_sp.shape[0]), min(mid_y + i,img_sp.shape[1])
        bbox = img_sp[mid_x_m:mid_x_p, mid_y_m:mid_y_p]
        index_tmp = np.unique(bbox)
        for k in index_tmp:
            if k not in index_io:
                index_io.append(k) 
    return index_io


def recon_vol(vol_sp):
    mid_x, mid_y, mid_z = int(vol_sp.shape[0]/2),int(vol_sp.shape[1]/2),int(vol_sp.shape[2]/2)
    index_io = []
    for i in range(0,max(mid_x, mid_y, mid_z)):
        mid_x_m, mid_y_m, mid_z_m = max(mid_x - i,0), max(mid_y - i,0), max(mid_z - i,0)
        mid_x_p, mid_y_p, mid_z_p = min(mid_x + i,vol_sp.shape[0]), min(mid_y + i,vol_sp.shape[1]), min(mid_z + i,vol_sp.shape[2])
        bbox = vol_sp[mid_x_m:mid_x_p, mid_y_m:mid_y_p, mid_z_m:mid_z_p]
        index_tmp = np.unique(bbox)
        for k in index_tmp:
            if k not in index_io:
                index_io.append(k) 
    return index_io