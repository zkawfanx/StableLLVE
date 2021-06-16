import argparse
import numpy as np
import torch
import torch.nn as nn
import cv2
from model import UNet
from glob import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./checkpoint.pth', help="path to the saved checkpoint of model")
args = parser.parse_args()

filenames = glob('./data/test/*')
filenames.sort()

model = UNet(n_channels=3, bilinear=True)
model.load_state_dict(torch.load(args.path))
model.to('cuda')

with torch.no_grad():
    for i, filename in enumerate(filenames):
        test = cv2.imread(filename)/255.0        
        test = np.expand_dims(test.transpose([2,0,1]), axis=0)
        test = torch.from_numpy(test).to(device="cuda", dtype=torch.float32)

        out = model(test)

        out = out.to(device="cpu").numpy().squeeze()
        out = np.clip(out*255.0, 0, 255)

        path = filename.replace('/test/','/results/')[:-4]+'.png'
        # folder = os.path.dirname(path)
        # if not os.path.exists(folder):
        #     os.makedirs(folder)
        cv2.imwrite(path, out.astype(np.uint8).transpose([1,2,0]))
        print('%d|%d'%(i+1, len(filenames)))
