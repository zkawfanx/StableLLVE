import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
import cv2
from glob import glob

class llenDataset(Dataset):
    def __init__(self, rootdir, type='train'):
        self.rootdir = rootdir
        self.type = type

        self.filepaths = self.get_filepath(rootdir, self.type)
        if self.type == 'train':
            self.transform = self.train_transform
        # else:
        #     self.transform = self.test_transform

    def get_filepath(self, rootdir, type):
        paths = []
        if self.type == 'train':
            filenames = glob(rootdir+'/train/*/*.png')
            for f in filenames:
                inputpath = f
                gtpath = f.replace('/train/', '/gt/')[:-4]+'.jpg'
                flowpath = f.replace('/train/', '/flow/')[:-4]+'.npy'
                
                paths.append((inputpath, gtpath, flowpath))
        # else:
        #     for f in filenames:
        #         inputpath = f
        #         gtpath = f.replace('/val/', '/gt/')[:-4]+'.jpg'
                
        #         paths.append((inputpath, gtpath))

        return paths
    
    def train_transform(self, lowlights, gts, flows):
        # Random crop
        i, j, h, w = self.getCropPosistion(lowlights, output_size=(512, 512))
        lowlights = lowlights[i:i+h, j:j+w]
        gts = gts[i:i+h, j:j+w]
        flows = flows[i:i+h, j:j+w]
        
        # Random rotation
        times = random.randint(0, 3)
        lowlights = np.rot90(lowlights, k=times)
        gts = np.rot90(gts, k=times)
        flows = np.rot90(flows, k=times)

        # Random horizontal flipping
        if random.random() > 0.5:
            lowlights = cv2.flip(lowlights, 1)
            gts = cv2.flip(gts, 1)
            flows = np.flip(flows, 1)
        
        # Random jitter on flow
        #x = random.uniform(-1,1)
        #y = random.uniform(-1,1)
        #flows[:,:,0] += x
        #flows[:,:,1] += y

        lowlights = (lowlights/255.0).astype(np.float32).transpose([2,0,1])
        gts = (gts/255.0).astype(np.float32).transpose([2,0,1])
        flows = flows.astype(np.float32).transpose([2,0,1])
        
        return lowlights, gts, flows
                

    # def test_transform(self, lowlights, gts):

    #     lowlights = (lowlights/255.0).astype(np.float32).transpose([2,0,1])
    #     gts = (gts/255.0).astype(np.float32).transpose([2,0,1])

    #     return lowlights, gts
    
    def __getitem__(self, index):
        if self.type == 'train':
            inputpath, gtpath, flowpath = self.filepaths[index]
            lowlight = cv2.imread(inputpath)
            gt = cv2.imread(gtpath)
            flow = np.load(flowpath)

            lowlight, gt, flow = self.transform(lowlight, gt, flow)
            return lowlight, gt, flow
        # else:
        #     inputpath, gtpath = self.filepaths[index]
        #     lowlight = cv2.imread(inputpath)
        #     gt = cv2.imread(gtpath)

        #     lowlight, gt= self.transform(lowlight, gt)
        #     return lowlight, gt
    
    def __len__(self):
        return len(self.filepaths)
    
    def getCropPosistion(self, input, output_size):
        h, w = input.shape[:2]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, th, tw

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw
