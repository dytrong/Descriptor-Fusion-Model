import numpy as np
import torch.nn as nn
from forward import *
import torchvision.models as models
import time

#####download models######
def select_pretrained_cnn_model(flag):
    start=time.time()
    if flag == 'alexnet':
        mynet = models.alexnet(pretrained = True)
    if flag == 'vgg16':
        mynet = models.vgg16(pretrained = True)
    if flag == 'resnet101':
        mynet = models.resnet101(pretrained = True)
    if flag == 'densenet169':
        mynet = models.densenet169(pretrained = True)
    mynet = mynet.cuda()
    mynet.eval()
    print('init spend time '+str(time.time() - start))
    return mynet

#####class#########
class generate_des:

    def __init__(self, net, img_tensor, mini_batch_size, net_type):

        self.descriptor = self.extract_batch_conv_features(net,img_tensor, mini_batch_size, net_type)

    #####extract batch conv features#####
    def extract_batch_conv_features(self, net, input_data, mini_batch_size, net_type):
        batch_number = len(input_data)//mini_batch_size
        #####计算第一个块的卷积特征
        descriptor = self.extract_conv_features(net, input_data[:mini_batch_size], net_type).cpu().detach().numpy()
        for i in range(1,batch_number):
            if i < batch_number-1:
                mini_batch = input_data[mini_batch_size*i:mini_batch_size*(i+1)]
            #######计算最后一个块,大于等于mini_batch_size
            if i == batch_number-1:
                mini_batch = input_data[mini_batch_size*i:len(input_data)]
            temp_descriptor = self.extract_conv_features(net, mini_batch, net_type).cpu().detach().numpy()
            #####np.vstack纵向拼接，np.hstack横向拼接
            descriptor = np.vstack((descriptor, temp_descriptor))
        return descriptor

    #####extract conv features#####
    def extract_conv_features(self, net, input_data, net_type):
        if net_type.startswith('alexnet'):
            x = alexnet(net, input_data)
        if net_type.startswith('vgg16'):
            x = vgg16(net, input_data)
        if net_type.startswith('vgg19'):
            x = vgg19(net, input_data)
        if net_type.startswith('inception_v3'):
            x = inception_v3(net, input_data)
        if net_type.startswith('resnet'):
            x = resnet(net, input_data)
        if net_type.startswith('densenet'):
            x = densenet(net, input_data)
        return x
