import cv2
import numpy as np
import os
import h5py

######提取图像的sift特征点
def sift_detect(img, Max_kp_num):

    #change the image format to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #extract sift features
    sift = cv2.xfeatures2d.SIFT_create(Max_kp_num)

    #detect the image keypoints
    keypoints = sift.detect(gray, None)

    return keypoints

######计算有效的特征点块，因为有些特征点在图像边缘地方，以该点为中心，取特征点块可能超过了图像的边界，要舍弃改特征点.
####Image_path:要匹配图像的路径
####Max_kp_num:检测的sift特征点最多个数
def compute_valid_keypoints(Image_path, Max_kp_num):

    img = cv2.imread(Image_path)

    keypoints = sift_detect(img, Max_kp_num)

    #generate the pathes based on keypoints  
    cv_pts, patch_img = generate_keypoints_patch(keypoints, img)

    return cv_pts, patch_img

#####计算特征点所在的图像金字塔层数
def unpack_octave(octave):

    octave = octave&255

    octave = octave if octave<128 else (-128|octave)

    return octave

###keypoints:extract from image by sift
###img:the original image
def generate_keypoints_patch(keypoints, img):

    patch_size_list = (32,32,32,32,32,32,32,32,32)

    cv_pts = []

    patch_img_list = []

    for k in keypoints:

        #because the image axis is defferent from matrix axis
        x = int(k.pt[1])

        y = int(k.pt[0])

        #unpack the octave
        size_index = unpack_octave(k.octave)+1

        patch_size = patch_size_list[size_index]

        #judgment the boundary
        if (x-patch_size)>0 and (y-patch_size)>0 and (x+patch_size)<img.shape[0] and (y+patch_size)<img.shape[1] and (k.pt not in cv_pts):

            #delete the same keypoints
            cv_pts.append(k.pt)

            #the image of keypoint field
            patch_img = img[x-patch_size:x+patch_size, y-patch_size:y+patch_size]

            patch_img_list.append(patch_img)

    #返回 valid different keypoints
    return cv_pts, patch_img_list

