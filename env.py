import random
import copy
import os
import gym
from gym.utils import seeding
import numpy as np
import math
from numpy.random import f
from scipy.spatial.distance import dice
import torch
import math
import nibabel as nib
from seg_eva import *
from scipy.stats import wasserstein_distance
import torch.nn.functional as F
from recon_env import *


def padding(array, xx, yy, zz):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]
    d = array.shape[2]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    c = (zz-d) // 2
    cc = zz - c - d
    return np.pad(array, pad_width=((a, aa), (b, bb), (c, cc)), mode='constant')


class RenderEnv(gym.Env):
    def __init__(self, path, args, is_train, classification):
        self.path = path
        self.is_train = is_train
        # loading data and annotation file
        self.vol, self.vol_sp, self.index, self.bg, self.info = self.load_data()
        self.x1, self.x2, self.y1, self.y2, self.z1, self.z2 = self.info
        self.vol_bbox = self.vol[self.x1:self.x2,self.y1:self.y2,self.z1:self.z2]
        self.bg_bbox = self.bg[self.x1:self.x2,self.y1:self.y2,self.z1:self.z2]
        self.current_vol = copy.copy(self.vol)
        self.segmap = np.zeros([self.vol_bbox.shape[0], self.vol_bbox.shape[1],self.vol_bbox.shape[2]])
        
        self.x, self.y, self.z = self.x1, self.y1, self.z1
        self.state_size = 32
        self.round = 0
        self.count_idx = 0
        current_index = self.index[self.count_idx]
        
        index_ = self.vol_sp == current_index
        self.current_x = np.where(index_ == 1)[0][math.floor(len(np.where(index_ == 1)[0]) / 2)] + self.x
        self.current_y = np.where(index_ == 1)[1][math.floor(len(np.where(index_ == 1)[1]) / 2)] + self.y
        self.current_z = np.where(index_ == 1)[2][math.floor(len(np.where(index_ == 1)[1]) / 2)] + self.z
        
        self.x0, self.y0, self.z0 = self.current_x, self.current_y, self.current_z
        
        self.current_state = copy.copy(self.current_vol[
                                    max(self.x0 - self.state_size,0):min(self.x0 + self.state_size,self.vol.shape[0]),
                                    max(self.y0 - self.state_size,0):min(self.y0 + self.state_size,self.vol.shape[1]),
                                    max(self.z0 - self.state_size,0):min(self.z0 + self.state_size,self.vol.shape[2])
                                    ])
        self.current_state = padding(self.current_state, self.state_size*2, self.state_size*2, self.state_size*2)
        
        self.termination = False
        self.theta = 25
        self.round = 1
        
    def step(self, action):
        self.take_action(action)
        reward = self._get_reward()
        vol, state = self._get_state()
        info = {
            "current_state": [self.current_state],
            "current_vol": [self.current_vol],
            "reward": reward,
        }
        return vol, state, reward, self.termination, info, self.segmap
    
    
    def take_action(self, action):
        ex_vol = copy.copy(self.current_vol)
        current_index = self.index[self.count_idx]
        index_ = self.vol_sp == current_index
        
        if action == 0:
            self.idr1 = 0
            self.idr2 = 0
            self.csr = 0
            pass
        elif action == 1:
            self.current_vol[self.x1:self.x2,self.y1:self.y2,self.z1:self.z2][index_] = self.bg_bbox[index_]
            
            pre_fg_list = (self.segmap*self.vol_bbox).tolist()
            pre_bg_list = ((1-self.segmap)*self.vol_bbox).tolist()
            
            self.segmap[index_] = 1
              
            after_fg_list = (self.segmap*self.vol_bbox).tolist()
            after_bg_list = ((1-self.segmap)*self.vol_bbox).tolist()
            
            wd1 = wasserstein_distance(pre_fg_list,after_fg_list)
            wd2 = wasserstein_distance(pre_fg_list,pre_bg_list)
            wd3 = wasserstein_distance(after_fg_list,after_bg_list)
            self.idr1 = wd1
            self.idr2 = wd3-wd2
            
            with torch.no_grad():
                ex_vol_ = ex_vol[self.x1:self.x2,self.y1:self.y2,self.z1:self.z2]/255.
                current_vol_ = self.current_vol[self.x1:self.x2,self.y1:self.y2,self.z1:self.z2]/255.
                
                cls_ex1 = torch.sigmoid(self.model(torch.tensor(ex_vol_[np.newaxis,np.newaxis,]).float().cuda()))[0][1]
                cls_current1 = torch.sigmoid(self.model(torch.tensor(current_vol_[np.newaxis,np.newaxis,]).float().cuda()))[0][1]

                if cls_current1<0.01:
                    self.termination = True
                
            self.csr = cls_ex1 - cls_current1
    
        self.count_idx += 1
        current_index = self.index[self.count_idx]
        index_ = self.vol_sp == current_index
        self.current_x = np.where(index_ == 1)[0][math.floor(len(np.where(index_ == 1)[0]) / 2)] + self.x
        self.current_y = np.where(index_ == 1)[1][math.floor(len(np.where(index_ == 1)[1]) / 2)] + self.y
        self.current_z = np.where(index_ == 1)[1][math.floor(len(np.where(index_ == 1)[1]) / 2)] + self.z
        self.x0, self.y0, self.z0 = self.current_x, self.current_y, self.current_z
        
        self.next_state = self.current_vol[
                                        max(self.x0 - self.state_size,0):min(self.x0 + self.state_size,self.vol.shape[0]),
                                        max(self.y0 - self.state_size,0):min(self.y0 + self.state_size,self.vol.shape[1]),
                                        max(self.z0 - self.state_size,0):min(self.z0 + self.state_size,self.vol.shape[2])
                                    ]
        self.next_state = padding(self.current_state, self.state_size*2, self.state_size*2, self.state_size*2)
        
        if self.count_idx == len(self.index)-1 and self.round == 1:
            self.count_idx = 0
            self.round += 1
        if self.count_idx == len(self.index)-1 and self.round == 2:
            self.termination = True
    
    def _get_reward(self):
        reward1 = np.sign(self.csr)
        if 0< self.idr1 < self.theta:
            reward2 = 1
        elif self.idr1 > self.theta:
            reward2 = -1
        reward3 = np.sign(self.idr2)
        reward_ = reward1 + reward2 + reward3
        return reward_
    
    def _get_state(self):
        return self.current_vol, self.next_state
    
    def load_data(self):
        # self.vol, self.vol_sp, self.index, self.bg, self.info
        vol = nib.load('path/to/nii').get_fdata()
        vol_sp = nib.load('path/to/vol_sp').get_fdata()
        recon_index = recon_vol(vol_sp)
        bg = nib.load('path/to/eraser_source').get_fdata()
        nodule_info = np.load('path/to/box',allow_pickle=True)[()]
        return vol ,vol_sp, recon_index, bg, nodule_info