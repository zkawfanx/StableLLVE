import torch, torchvision
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import os
from bitarray import bitarray
import argparse
from glob import glob

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


# Select a model and its config file from the model zoo
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Cityscapes/mask_rcnn_R_50_FPN.yaml")

print('....model loading....')
predictor = DefaultPredictor(cfg)
print('model loaded')


# Put clean images you want to predict masks for here
rootdir = './examples/'
folders = os.listdir(os.path.join(rootdir, 'gt'))
folders.sort()

for f in folders:
    path = os.path.join(rootdir, 'mask', f) # Folder for Mask files
    if not os.path.exists(path):
        os.makedirs(path)
    
    path = os.path.join(rootdir, 'mask_img', f) # Folder for Visualizations
    if not os.path.exists(path):
        os.makedirs(path)

filepaths = []
for f in folders:
    filepaths.extend(glob(os.path.join(rootdir, 'gt', f, '*.jpg')))

filepaths.sort()
for i in range(len(filepaths)):
    filepath = filepaths[i]
    image = cv2.imread(filepath)

    # Instance Segmentation using Detectron2 Model
    outputs = predictor(image)

    # Obtain Class_ids, Masks and Confidence from Outputs
    r = outputs['instances'].to('cpu')
    ids = r.pred_classes
    masks = r.pred_masks
    scores = r.scores

    # Filter out instances with a confidence lower than your threshold, e.g., 85%
    ids = ids[scores>0.85]
    masks = masks[scores>0.85].numpy().transpose([1,2,0])
    scores = scores[scores>0.85]


    ''' 
    # You can also obtain instances of classes you preferred
    # You can find these predefined classes in detectron2/data/datasets/bultin_meta.py
    # Optional

    N = ids.shape[0]
    if not N:
        index = np.array([],dtype=np.uint8)
    else:
        assert ids.shape[0] == masks.shape[0]
        index = np.concatenate([np.where(ids==0)[0], np.where(ids==2)[0],\
                                np.where(ids==1)[0], np.where(ids==7)[0]])

    masks = masks[index, :, :].transpose([1,2,0])
    '''


    # Save mask files
    maskpath = filepath.replace('/gt/','/mask/')[:-4]+'.npy'  
    np.save(maskpath, masks)

    '''
    # Since mask files are all 0/1 bitmaps, you can encode it into bitarray for save of storage
    # Optional

    m = masks.tobytes()
    b = bitarray()
    b.pack(m)
    path = os.path.join(rootdir, 'mask', filename[:-4]+'.bin')
    with open(path, 'wb') as f:
        b.tofile(f)
    '''

    # Save visualizations of instance segmentation results
    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
    out = v.draw_instance_predictions(r)
    #cv2.imwrite(os.path.join(rootdir, 'masked_img', fn[:-4]+'.jpg'), out.get_image()[:, :, ::-1])
    cv2.imwrite(filepath.replace('/gt/','/mask_img/')[:-4]+'.jpg', out.get_image()[:, :, ::-1])
    print(str(i+1)+' completed | total %d'%len(filepaths))