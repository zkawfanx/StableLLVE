# Optical Flow Prediction
To predict feasible optical flow for clean images, you need to get instance segmentation masks for each foreground object first. Then random vectors are sampled on areas of masks to predict optical flow. **[Detectron2](https://github.com/facebookresearch/detectron2)** and **[CMP](https://github.com/XiaohangZhan/conditional-motion-propagation)** are used to achieve this goal.

## Instance Segmentation with Detectron2
We choose **[Detectron2](https://github.com/facebookresearch/detectron2)** for instance segmentation in the paper. Installation for it and related dependencies can be found **[here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)**. A simple prediction code is provided and you can get masks by running
```shell
cd ./etc
python predict_mask.py 
```
Note: The version of detectron2 used here is **[v0.3](https://github.com/facebookresearch/detectron2/releases/tag/v0.3)**. You can try newer version and adapt the code accordingly.

If you use Detectron2 in your research, please cite their paper.
```
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```

## Predict Your Optical Flow with CMP
With masks of foreground objects, you can predict optical flow using pretrained model of **[CMP](https://github.com/XiaohangZhan/conditional-motion-propagation)**. Since its demo is a jupter notebook which requires human interactions, we adapted the code into a naive automatic version and replace the inputs with randomly sampled vectors on areas of instance masks. Try its original demo and find out how it actually works with different inputs. You shall install required dependencies and download pretrained model from original repo. A simple prediction code is provided and you can predict your optical flow by running
```shell
cd ./etc
python predict_optical_flow.py 
```
If you use CMP in your research, please cite their paper.
```
@inproceedings{zhan2019self,
 author = {Zhan, Xiaohang and Pan, Xingang and Liu, Ziwei and Lin, Dahua and Loy, Chen Change},
 title = {Self-Supervised Learning via Conditional Motion Propagation},
 booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR)},
 month = {June},
 year = {2019}
}
```

## Example
For example, if you input an image in the `examples/gt/dance-twirl` folder,

![](examples/gt/dance-twirl/00000.jpg)

an instance segmentation result can be obtained using detectron2, the visualization can be found in the `examples/mask_img/dance-twirl` folder,

![](examples/mask_img/dance-twirl/00000.jpg)

an optical flow can be further obtained using CMP and its visualization is saved in `examples/flow_img/dance-twirl` folder.

![](examples/flow_img/dance-twirl/00000.jpg)