# Pixel-wise Warping for Video Stabilization (PWNet)
This is a [PyTorch](http://pytorch.org/) implementation of Pixel-wise Warping for Video Stabilization.

Source code and models will be opened soon!

If you have any questions, please contact with me:
zmd1992@mail.ustc.edu.cn

### Table of Contents
- <a href='#Prerequisites'>Prerequisites</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training-ssd'>Train</a>
- <a href='#performance'>Performance</a>
- <a href='#demos'>Demos</a>
- <a href='#references'>Reference</a>

&nbsp;
&nbsp;
&nbsp;
&nbsp;

## Prerequisites
- Linux
- Python 3
- NVIDIA GPU (12G or 24G memory) + CUDA cuDNN
- pytorch 0.4.0+
- numpy
- cv2
- ...

## Datasets
The dataset for is the DeepStab dataset (7.9GB) http://cg.cs.tsinghua.edu.cn/download/DeepStab.zip thanks to Miao Wang [1]. 

## Training 
- The code will download the [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at:             https://download.pytorch.org/models/vgg16-397923af.pth automatically.

- To train PWNet using the train script simply specify the parameters listed in `./lib/cfg.py` as a flag or manually change them.
- The default parameters are set for the use of two NVIDIA 1080Ti graphic cards with 24G memory.

```Shell
CUDA_VISIBLE_DEVICES=0,1 python3 main.py
```

- Note:
  * For training, an NVIDIA GPU is strongly recommended for speed.
  * Before training, you should ensure the location of preprocessed dataset, which will be supplied soon.


## Performance

Detailed performance can be seen in our paper.
- Pixel-wise Warping for Video Stabilization with
Deep Generative Adversarial Networks

- Here we show a example videos to compare our PWNet with StabNet [1]
  

<video id="video" controls="" preload="none" poster="http://om2bks7xs.bkt.clouddn.com/2017-08-26-Markdown-Advance-Video.jpg"> <source id="avi" src="https://github.com/mindazhao/pix-pix-warping-video-stabilization/example.avi" type="video/avi"> </video> 



## Demos

### Use a pre-trained PWNet for video stabilization

#### Download a pre-trained network
- We are trying to provide a pre-trained model.
- Currently, we provide the following PyTorch models:
       model wil be opened soon!
- You can test your own unstable videos by changing the parameter "train" with False and adjust the path yourself in function "process()".
    




## Authors
- Minda Zhao

## References
[1] M. Wang, G.-Y. Yang, J.-K. Lin, S.-H. Zhang, A. Shamir, S.-P. Lu,
and S.-M. Hu, “Deep online video stabilization with multi-grid warp-
ing transformation learning,” IEEE Transactions on Image Processing,
vol. 28, no. 5, pp. 2283–2292, 2019