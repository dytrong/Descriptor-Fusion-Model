# Descriptor-Fusion-Model (DFM)

![image](https://github.com/dytrong/Descriptor-Fusion-Model/blob/master/main/log/system_fusion.jpg)


Fig. 1: The details of our descriptor fusion model (DFM). The descriptor fusion steps are as follows. First, we use a keypoint
detector (SIFT in our experiments) to detect K keypoints and then extract 64 × 64 image patches around these keypoints.
Secondly, each of the image patches is fed into a trained CNN model and a pre-trained CNN model to generate two keypoint
descripors, for all the K keypoints. Subsequently, we use a convolutional auto-encoder to compress the descriptor from the
pre-trained model. Finally the descriptor from the trained CNN model and the compressed pre-trained descriptor are fused
by a fully-connected autoencoder. For either autoencoder, it is trained with the standard reconstruction loss and, after training
only the encoder half is used.

# Pre-Trained autoencoder model (CAE and FCAE)

The pre-trained CAE and FCAE parameters can get by https://pan.baidu.com/s/1LyNmDhXq2EntqXQJOJy36w and the password is 2vmr. The pre-trained CAE parameters you can place it in "./CAE/parameters/" and the pre-trained FCAE parameters places in "./FCAE/model_parameters/".

# Requirements

1. Pytorch 0.4.0

2. Opencv 3.1 and opencv_contrib 

# Usage example
```
cd main/

sh DFM.sh
```
Our DFM descriptor results can be seen in log file.

# Compared with state-of-the-art descriptors

![image](https://github.com/dytrong/Descriptor-Fusion-Model/blob/master/main/log/ours.jpg)

Fig. 2: Qualitative comparison of five descriptors for keypoints matching. From top to bottom: SIFT [1], GeoDesc [2],
ContextDesc [3], HardNet [4] and our DFM. The first two columns of images are from illumination sequences of Hpatches
and the last two columns of images are viewpoint sequences. For the convenience of viewing, we only show the 20 pairs
of matching keypoints with the highest similarity for each image pair. The red lines show incorrect matches and the blue
lines the correct matches. Our DFM descriptor achieves the best results in both the illumination sequences and viewpoint
sequences.


# References
[1] D. G. Lowe, “Distinctive image features from scale-invariant keypoints,” International journal of computer vision, vol. 60, no. 2, pp. 91–110, 2004.

[2] Z. Luo, T. Shen, L. Zhou, S. Zhu, R. Zhang, Y. Yao, T. Fang, and L. Quan, “Geodesc: Learning local descriptors by integrating geometry constraints,” in Proceedings of the European Conference on Computer Vision (ECCV), 2018, pp. 168–183.

[3] Z. Luo, T. Shen, L. Zhou, J. Zhang, Y. Yao, S. Li, T. Fang, and L. Quan, “Contextdesc: Local descriptor augmentation with cross-modality context,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019, pp. 2527–2536.

[4]A. Mishchuk, D. Mishkin, F. Radenovic, and J. Matas, “Working hard to know your neighbor’s margins: Local descriptor learning loss,” in Advances in Neural Information Processing Systems, 2017, pp. 4826–4837.
