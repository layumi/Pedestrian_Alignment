# Pedestrian Alignment Network for Person Re-identification

This repo is for our IEEE TCSVT paper (https://arxiv.org/abs/1707.00408). 
The main idea is to align the pedestrian within the bboxes, and reduce the noisy factors, i.e., scale and pose variances.

## Network Structure
![](https://github.com/layumi/Pedestrian_Alignment/blob/master/fig2.jpg)
For more details, you can see this [png file](https://raw.githubusercontent.com/layumi/Pedestrian_Alignment/master/PAN.png). But it is low-solution now, and I may replace it recently.

## Installation
1.Clone this repo.

    git clone https://github.com/layumi/Pedestrian_Alignment.git
    cd Pedestrian_Alignment
    mkdir data

2.Download the pre-trained model. Put it into './data'.

    cd data
    wget http://www.vlfeat.org/matconvnet/models/imagenet-resnet-50-dag.mat
    
3.Compile Matconvnet
**(Note that I have included my Matconvnet in this repo, so you do not need to download it again. I have changed some codes comparing with the original version. For example, one of the difference is in `/matlab/+dagnn/@DagNN/initParams.m`. If one layer has params, I will not initialize it again, especially for pretrained model.)**

You just need to uncomment and modify some lines in `gpu_compile.m` and run it in Matlab. Try it~
(The code does not support cudnn 6.0. You may just turn off the Enablecudnn or try cudnn5.1)

If you fail in compilation, you may refer to http://www.vlfeat.org/matconvnet/install/
    
## Dataset
Download [Market1501 Dataset] (http://www.liangzheng.org/Project/project_reid.html)

For training CUHK03, we follow the new evaluation protocol in the [CVPR2017 paper](https://github.com/zhunzhong07/person-re-ranking). It conducts a multi-shot person re-ID evaluation and only needs to run one time.

## Train
1. Add your dataset path into `prepare_data.m` and run it. Make sure the code outputs the right image path.

2. uncomment https://github.com/layumi/Pedestrian_Alignment/blob/master/resnet52_market.m#L23 

Run `train_id_net_res_market_new.m` to pretrain the base branch.

3. comment https://github.com/layumi/Pedestrian_Alignment/blob/master/resnet52_market.m#L23 

Run `train_id_net_res_market_align.m` to finetune the whole net.

## Test
1. Run `test/test_gallery_stn_base.m` and `test/test_gallery_stn_align.m` to extract the image features from base branch and alignment brach. Note that you need to change the dir path in the code. They will store in a .mat file. Then you can use it to do the evaluation.

2. Evaluate feature on the Market-1501. Run `evaluation/zzd_evaluation_res_faster.m`. You can get a Single-query Result around the following result.

| Methods               | Rank@1 | mAP    | 
| --------              | -----  | ----   | 
| Ours           | 82.81% | 63.35% | 

You may find our trained model at [GoogleDrive](https://drive.google.com/open?id=1X09jnURIicQk7ivHjVkq55NHPB86hQT0)
## Visualize Results
We conduct an extra interesting experiment:
**When zooming in the input image (adding scale variance), how does our alignment network react?**

We can observe a robust transform on the output image (focusing on the human body and keeping the scale).

The left image is the input; The right image is the output of our network.

![](https://github.com/layumi/Person_re-ID_stn/blob/master/gif/0018_c4s1_002351_02_zoomin.gif)
    ![](https://github.com/layumi/Person_re-ID_stn/blob/master/gif/0153_c4s1_026076_03_zoomin.gif)
    ![](https://github.com/layumi/Pedestrian_Alignment/blob/master/gif/0520_c4s3_001373_03_zoomin.gif)


![](https://github.com/layumi/Pedestrian_Alignment/blob/master/gif/0520_c5s1_143995_06_zoomin.gif)
    ![](https://github.com/layumi/Pedestrian_Alignment/blob/master/gif/0345_c6s1_079326_07_zoomin.gif)
    ![](https://github.com/layumi/Pedestrian_Alignment/blob/master/gif/0153_c4s1_025451_01_zoomin.gif)

## Citation
Please cite this paper in your publications if it helps your research:
```
@article{zheng2017pedestrian,
  title={Pedestrian Alignment Network for Large-scale Person Re-identification},
  author={Zheng, Zhedong and Zheng, Liang and Yang, Yi},
  doi={10.1109/TCSVT.2018.2873599},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2018}
}
```

## Acknowledge
Thanks for the suggestions from Qiule Sun.
