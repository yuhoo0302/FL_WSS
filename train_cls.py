import os
import argparse
import json
import numpy as np
from abus_dataset import ABUSLoader
from bus_dataset import BUSLoader
from cls_model import *
from torch import optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR


def train_model(model, criterion, train_dataloaders, val_dataloaders, num_epochs=150):
    test_acc = []
    optimizer = optim.Adam(model.parameters(),lr=1e-4)
    scheduler = StepLR(optimizer, step_size=60, gamma=0.1)
    for epoch in np.arange(num_epochs) + 1:
        model.train()
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        dt_size = len(train_dataloaders.dataset)
        epoch_loss = 0
        step = 0

        for x, y in train_dataloaders:
            step += 1
            inputs = x.cuda()
            label = y.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if step % 20 ==0:
                print("%d/%d,train_loss:%0.8f" % (step, (dt_size - 1) // train_dataloaders.batch_size + 1, loss.item()))
        model.eval()
        acc_sum = []
        loss_val = 0.
        with torch.no_grad():
            for y_pre, gt in val_dataloaders:  
                pre = model(y_pre.cuda().float())
                loss_ = criterion(pre, gt.cuda())
                
                pre = np.argmax(pre.cpu().numpy(),axis=1)
                gt = gt.numpy().ravel()
                acc_num = sum(pre == gt)
                acc_sum.append(acc_num)
                loss_val += loss_.item()
            
            loss_val = loss_val / len(val_dataloaders.dataset)
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Current Learning Rate: {current_lr}')
            
            acc = sum(acc_sum)/len(val_dataloaders.dataset)
            test_acc.append(acc)
            
            print('weights_test{}, current_acc = {:.5f}, best_acc = {:.5f}, val_loss = {:.5f}'.format(epoch, test_acc[epoch - 1], max(test_acc), loss_val))
            if test_acc[epoch - 1] == max(test_acc):
                print('The epoch {} is the temp optimal model!'.format(epoch))
            save_path = '.path/to/save'
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), '{}/weights_{:.2f}_{}.pth'.format(save_path, acc, epoch))
    return model


def train():
    if args.mode == '3d':
        model = resnet18_3d(num_classes=2).cuda()
        train_dataset = ABUSLoader(option='train')
        val_dataset = ABUSLoader(option = 'val')
    elif args.mode == '2d':
        model = resnet18_2d(num_classes=2).cuda()
        train_dataset = BUSLoader(option='train')
        val_dataset = BUSLoader(option = 'val')
    
    batch_size = args.batch_size
    print('#batch_size:{}'.format(batch_size))
    criterion = torch.nn.CrossEntropyLoss()
    
    train_dataloaders = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8) 
    val_dataloaders = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    train_model(model, criterion, train_dataloaders, val_dataloaders)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("mode", type=str, default='2d', help="2d or 3d")
    parse.add_argument("--batch_size", type=int, default=8)
    parse.add_argument("--ckp", type=str, help="the path of model weight file")
    parse.add_argument('--reuse', default=None, help='model state path to load for reuse')
    args = parse.parse_args()
    torch.cuda.set_device(1)   
    if args.action == "train":
        train()
