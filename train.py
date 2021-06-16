import argparse
import os, socket
from datetime import datetime
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from model import UNet
from warp import WarpingLayerBWFlow

from torch.utils.tensorboard import SummaryWriter
from dataloader import llenDataset
from torch.utils.data import DataLoader

import cv2
import kornia
import random

def save_checkpoint(state, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth')
    torch.save(state, checkpoint_filename)

# Parse arguments
parser = argparse.ArgumentParser(description='Low light enhancement')
parser.add_argument('--data-path', default='./data', type=str, help='path to the dataset')
parser.add_argument('--epochs', default=50, type=int, help='n of epochs (default: 50)')
parser.add_argument('--bs', default=1, type=int, help='[train] batch size(default: 1)')
parser.add_argument('--bs-test', default=1, type=int, help='[test] batch size (default: 1)')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate (default: 1e-4)')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use (default: 0)')
parser.add_argument('--checkpoint', default=None, type=str, help='path to checkpoint')
parser.add_argument('--log', default=None, type=str, help='folder to log')
parser.add_argument('--weight', default=20, type=float, help='weight of consistency loss')
args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
train_set = llenDataset(args.data_path, type='train')
train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True)

torch.manual_seed(ord('c')+137)
random.seed(ord('c')+137)
np.random.seed(ord('c')+137)

start_epoch = 0
model = UNet(n_channels=3, bilinear=True).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

criterion = nn.L1Loss()
warp = WarpingLayerBWFlow().cuda()

# Create logger
if args.log==None:
    log_dir = os.path.join(os.path.abspath(os.getcwd()), 'logs', datetime.now().strftime('%b%d_%H-%M-%S_') + socket.gethostname())
else:
    log_dir = os.path.join(os.path.abspath(os.getcwd()), 'logs', args.log)

os.makedirs(log_dir)
logger = SummaryWriter(log_dir)

# Log arguments
with open(os.path.join(log_dir, "config.txt"), "a") as f:
    print(args, file=f)

iters = 0
for epoch in range(start_epoch, args.epochs):
    # log learning rate
    for i, param_group in enumerate(optimizer.param_groups):
        logger.add_scalar('Lr/lr_' + str(i), float(param_group['lr']), epoch)


    # Training stage
    print('Epoch', epoch, 'train in progress...')
    model.train()

    for i, (input, target, flow) in enumerate(train_loader):
        input, target, flow= input.cuda(), target.cuda(), flow.cuda()

        # the 1st pass  
        pred = model(input)
        loss = criterion(pred, target)

        # the 2nd pass
        input_t = warp(input, flow)
        input_t_pred = model(input_t)
        pred_t = warp(pred, flow)
        
        loss_t = criterion(input_t_pred, pred_t)
        total_loss = loss + loss_t * args.weight
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        
        logger.add_scalar('Train/Loss', loss.item(), iters)
        logger.add_scalar('Train/Loss_t', loss_t.item(), iters)
        iters += 1

        if (i + 1) % 10 == 0:
            print('Train Epoch: {0} [{1}/{2}]\t'
                  'l1Loss={Loss1:.8f} '
                  'conLoss={Loss2:.8f} '.format(
                epoch, i + 1, len(train_loader), Loss1=loss.item(), Loss2=loss_t.item()))
    
    save_checkpoint(model.state_dict(), epoch, log_dir)
    print()
    
logger.close()