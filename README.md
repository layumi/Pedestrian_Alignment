# Pedestrian Alignment Network

This repo is for our arXiv paper (submitting). 
The main idea is to align the pedestrian within the bboxes, and reduce the noisy factors i.e., scale and pose variances.

# Visualize Results
We conduct an extra interesting experiment:
**When zooming in the input image (adding scale variance), how do our alignment network react?**

We can observe a robust transform on the output image (focusing on the human body and keeping the scale).

The left image is the input; The right image is the output of our network.

![](https://github.com/layumi/Person_re-ID_stn/blob/master/gif/0018_c4s1_002351_02_zoomin.gif)
    ![](https://github.com/layumi/Person_re-ID_stn/blob/master/gif/0153_c4s1_026076_03_zoomin.gif)
    ![](https://github.com/layumi/Pedestrian_Alignment/blob/master/gif/0520_c4s3_001373_03_zoomin.gif)


![](https://github.com/layumi/Pedestrian_Alignment/blob/master/gif/0520_c5s1_143995_06_zoomin.gif)
    ![](https://github.com/layumi/Pedestrian_Alignment/blob/master/gif/0345_c6s1_079326_07_zoomin.gif)
    ![](https://github.com/layumi/Pedestrian_Alignment/blob/master/gif/0153_c4s1_025451_01_zoomin.gif)



