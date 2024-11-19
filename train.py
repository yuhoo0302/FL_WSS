from __future__ import division
import copy
import os
from itertools import count
from env import RenderEnv
from agent import Agent, beta_by_frame
import random
import numpy as np
from cls_model import *
import argparse
import torch
import cv2
import glob

device = torch.device('cuda:0')

def main(args):

    agent = Agent(args)

    train_list = glob.glob(args.train_path)
    train_list = [os.path.join(args.train_path, d) for d in train_list]
    random.shuffle(train_list)

    """load the classification model"""
    cls_model = resnet18_2d(num_classes=2).to(device)
    cls_model.load_state_dict(torch.load(args.cls_model))
    cls_model.eval()
    
    for _ep in range(0, 0 + args.num_epoch):
        reward_train = []
        print('EPISODE :- ', _ep)
        random.shuffle(train_list)
        for train_id, train_path in enumerate(train_list):
            print(_ep, train_id, train_path)
            env = RenderEnv(train_path, args, is_train=True, classification=cls_model)
            current_state = env.get_state()
            state = np.concatenate((current_state[np.newaxis, ],
                                    current_state[np.newaxis, ],
                                    current_state[np.newaxis, ],
                                    ), axis=0)

            for t in count(1):
                i += 1
                agent.frame_idx += 1
                action = agent.select_action(state, random_action=True)
                _, new_state, reward, done, info, _ = env.step(action)
                reward_train.append(reward)
    
                next_state = np.concatenate((state[1: 3, :, :], new_state[np.newaxis,]))

                agent.replay_buffer.push(state, action, reward, next_state, done)
                state = copy.copy(next_state)

                print("\r none: {}, nodule: {} ".format(info['cls'][0], info['cls'][1]), end="")

                if i % args.learning_frequency == 0:
                    beta = beta_by_frame(agent.frame_idx)
                    agent.learn(beta)
                if done or info['cls'][0] > 0.99:
                    break
                
        torch.save(agent.eval_net.state_dict(), os.path.join(ckpt_path, "agent_{}.pth.gz".format(_ep)))


if __name__ == '__main__':
    class Parser(object):
        def __init__(self):
            self.data_enhance = False
            self.save = ''
            self.batch_size = 64
            self.lr = 5e-5
            self.weight_decay = 1e-4
            self.gamma = 0.9
            self.memory_capacity = 8000
            self.epsilon = 0.4
            self.learning_frequency = 1200
            self.gpu_id = 0
            self.num_epoch = 100 # 
            self.action_space = 2
            self.train_path = r'.path/to/train'
            self.output_path = r'path/to/output'
            self.source_path = r'path/to/source'
            self.cls_model = r'path/to/cls_model'
            self.target_step_counter = 6000
            
            self.ep_frequency = 10000

    parser = Parser()
    main(parser)