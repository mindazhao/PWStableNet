# PWStableNet: Learning Pixel-wise Warping Maps for Video Stabilization
This is a [PyTorch](http://pytorch.org/) implementation of PWStableNet: Learning Pixel-wise Warping Maps
for Video Stabilization.

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



### Original result before improvement:
we show an example videos to compare our PWNet with StabNet [1]
![image](https://github.com/mindazhao/pix-pix-warping-video-stabilization/blob/modified/videos/example.gif)
- The video can be download here.
https://github.com/mindazhao/pix-pix-warping-video-stabilization/blob/modified/videos/example.mp4
### More video result with improved PWStableNet 
- Example I: https://github.com/mindazhao/pix-pix-warping-video-stabilization/blob/modified/videos/example1.avi
- Example II: https://github.com/mindazhao/pix-pix-warping-video-stabilization/blob/modified/videos/example2.avi
- Parallax:
  1. weak parallax: https://github.com/mindazhao/pix-pix-warping-video-stabilization/blob/modified/videos/weak_parallax.avi
  2. middle parallax: https://github.com/mindazhao/pix-pix-warping-video-stabilization/blob/modified/videos/middle_parallax.avi
  3. strong parallax: https://github.com/mindazhao/pix-pix-warping-video-stabilization/blob/modified/videos/strong_parallax.avi
- Low quality: https://github.com/mindazhao/pix-pix-warping-video-stabilization/blob/modified/videos/Low quality.avi

### Note: If you have any problem to download these videos, you can visit another website: http://home.ustc.edu.cn/~zmd1992/PWStableNet.html



## Demos

### Use a pre-trained PWNet for video stabilization

#### Download a pre-trained network
- We are trying to provide a pre-trained model.
- Currently, we provide the following PyTorch models:
       model can be get from home.ustc.edu.cn/~zmd1992/PWStableNet/netG_model.pth
- You can test your own unstable videos by changing the parameter "train" with False and adjust the path yourself in function "process()".
    




## Authors
- Minda Zhao

## References
[1] M. Wang, G.-Y. Yang, J.-K. Lin, S.-H. Zhang, A. Shamir, S.-P. Lu,
and S.-M. Hu, “Deep online video stabilization with multi-grid warp-
ing transformation learning,” IEEE Transactions on Image Processing,
vol. 28, no. 5, pp. 2283–2292, 2019
