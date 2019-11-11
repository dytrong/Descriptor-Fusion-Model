import numpy as np
import cv2
import torch
from PIL import Image
import configparser
import h5py
import torchvision.transforms as transforms

#####change images to tensor#####
def change_patch_size(img_patch, model_size=224, img_channel=3):

    img_to_tensor = transforms.ToTensor()

    resized_seq = torch.zeros(len(img_patch), img_channel, model_size, model_size)

    for i in range(len(img_patch)):

        if img_channel == 1:

            img_patch[i] = cv2.cvtColor(img_patch[i], cv2.COLOR_BGR2GRAY)

            #img_patch[i] = (img_patch[i] - np.mean(img_patch[i])) / (1e-8 + np.std(img_patch[i]))

        tmp_patch = cv2.resize(img_patch[i], (model_size, model_size))

        tmp_patch = tmp_patch.reshape(model_size, model_size, img_channel)

        tmp_patch = img_to_tensor(tmp_patch)

        resized_seq[i] = tmp_patch

    return resized_seq


#######标准化
def normalization_desc(desc):

    mean_data = desc.mean(axis=1, keepdims=True)

    std_data = desc.std(axis=1, keepdims=True)

    desc -= mean_data

    desc /= std_data

    return desc


