import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import yaml
import os
import cv2
import random
#from bitarray import bitarray

from glob import glob

from utils import flowlib
import models
import utils
import importlib
importlib.reload(utils)


# load the parameters of model
class ArgObj(object):
    def __init__(self):
        pass

exp = './CMP/experiments/semiauto_annot/resnet50_vip+mpii_liteflow'
load_iter = 42000
configfn = "{}/config.yaml".format(exp)

args = ArgObj()
with open(configfn) as f:
    config = yaml.safe_load(f)

for k, v in config.items():
    setattr(args, k, v)

setattr(args, 'load_iter', load_iter)
setattr(args, 'exp_path', os.path.dirname(configfn))

model = models.__dict__[args.model['arch']](args.model, dist_model=False)
model.load_state("{}/checkpoints".format(args.exp_path), args.load_iter, False)
model.switch_to('eval')

data_mean = args.data['data_mean']
data_div = args.data['data_div']

img_transform = transforms.Compose([transforms.Normalize(data_mean, data_div)])
fuser = utils.Fuser(args.model['module']['nbins'], args.model['module']['fmax'])
torch.cuda.synchronize()




rootdir = './examples/'
folders = os.listdir(os.path.join(rootdir, 'gt'))
folders.sort()

for f in folders:    
    path = os.path.join(rootdir, 'flow', f) # Folder for optical flow predictions
    if not os.path.exists(path):
        os.makedirs(path)
    
    path = os.path.join(rootdir, 'flow_img', f) # Folder for visuliaztions
    if not os.path.exists(path):
        os.makedirs(path)


filepaths = []
for f in folders:
    filepaths.extend(glob(os.path.join(rootdir, 'gt', f, '*.jpg')))


filepaths.sort()
seg_path = [fp.replace('/gt/','/mask/')[:-4]+'.npy' for fp in filepaths]
# seg_path = [fp.replace('/gt/','/mask/')[:-4]+'.npy' for fp in filepaths]
flow_path = [fp.replace('/gt/','/flow/')[:-4]+'.npy' for fp in filepaths]
flowimg_path = [fp.replace('/gt/','/flow_img/')[:-4]+'.jpg' for fp in filepaths]


for i in range(len(filepaths)):
    # input an image and do some transform
    image = Image.open(filepaths[i]).convert('RGB')
    size = image.size

    '''
    # Decode the mask if you encode it when using detectron2
    # Optional 

    seg = bitarray()
    with open(seg_path[i], 'rb') as f:
       seg.fromfile(f)
    
    seg = seg.unpack(zero=b'\x00', one=b'\x01')
    seg = np.frombuffer(seg, dtype=np.uint8).reshape(size[1],size[0],-1)
    '''
    
    seg = np.load(seg_path[i])

    repeat = 1
    tensor = img_transform(torch.from_numpy(np.array(image).astype(np.float32).transpose((2,0,1))))
    image = tensor.unsqueeze(0).repeat(repeat,1,1,1).cuda()

    # Randomly sample vectors
    coords = []

    if seg.shape[0] != 0:
        n = seg.shape[2]
        for j in range(n):
            m = seg[:,:,j]
            (x, y) = np.where(m==1)
            xy = np.concatenate([np.expand_dims(x, axis=1),np.expand_dims(y,axis=1)],axis=1)

            # (u, v) points to a specific direction, with a certain length for the vector
            # 15 can be any other proper number
            u = random.randrange(-15,15)
            v = random.randrange(-15,15)

            # number of samples, the naive code contains no strategy to keep all samples uniformly located in the mask
            num_samples = 25
            for k in range(num_samples):
                r = random.choice(xy)            
                coords.append([r[1],r[0], u, v])
            
    sparse = np.zeros((1, 2, image.size(2), image.size(3)), dtype=np.float32)
    mask = np.zeros((1, 2, image.size(2), image.size(3)), dtype=np.float32)
    for arr in coords:
        sparse[0, :, int(arr[1]), int(arr[0])] = np.array(arr[2:4])
        mask[0, :, int(arr[1]), int(arr[0])] = np.array([1, 1])

    image = image.cuda()
    sparse = torch.from_numpy(sparse).cuda()
    mask = torch.from_numpy(mask).cuda()
    model.set_input(image, torch.cat([sparse, mask], dim=1), None)

    # inference for flow field
    tensor_dict = model.eval(ret_loss=False)
    flow = tensor_dict['flow_tensors'][0].cpu().numpy().squeeze().transpose(1,2,0).astype(np.float16)
    
    # clear artifacts out of masks
    tmp = np.sum(seg,axis=2)
    tmp = np.expand_dims((tmp>0), axis=2)
    tmp = np.concatenate([tmp,tmp],axis=2)
    flow = flow*tmp
    
    # save the results
    np.save(flow_path[i], flow)
    out = flowlib.flow_to_image(flow)
    out = Image.fromarray(out)
    out.save(flowimg_path[i])
    
    print('%d completed | %d in total'%(i+1,len(filepaths)))
    