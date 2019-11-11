from sklearn.decomposition import PCA
from sklearn import random_projection
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
import numpy as np
import time
import torch
import sys

"""
method: 降维的方法
dimension: 降到的维度
desc_path: 训练集的地址
auto_model_parameter: auto-encoder模型参数地址
pre_trained_model_name: 预训练模型的名称
"""
#####最后两个参数，只有使用Auto-Encoder的时候才需要用到
def select_dimension_reduce_model(method, dimension, desc_path, auto_model_parameter, pre_trained_model_name):
    start = time.time()
    ####深度学习的方法, Auto-Encoder
    if method == 'AUTO':
        sys.path.append('/home/zhuang/Fusions/Descriptor-Fusion-Model/CAE/')
        from train_CAE import autoencoder
        model = autoencoder(auto_model_parameter, pre_trained_model_name, dimension, pretrained=True)
        model = model.cuda()
        model.eval()
        return model
    ####传统的降维方法,训练数据集的数量不能小于降到的维度
    if method == 'PCA':
        train_data = np.load(desc_path)
        model = PCA(n_components = dimension)
    if method == 'RP':
        train_data = np.load(desc_path)
        model = random_projection.GaussianRandomProjection(n_components = dimension)
    train_model = model.fit(train_data)
    print('训练降维模型共耗时:' + str(time.time() - start))
    return train_model

"""
model: 训练好的模型
input_data: 输入新的数据(与训练数据格式一致)
method: 选择降维的方法
pre_trained_model_type: Auto-Encoder中预训练模型的选择
"""

def dimension_reduce_method(model, input_data, method, pre_trained_model_type, data_type = 'cuda'):
    ######传统的降维方法
    if method == 'PCA':
        ####model是训练好的模型,transform可以对新输入的数据进行降维
        out_desc = model.transform(input_data)
    if method == 'RP':
        out_desc = model.transform(input_data)
    if method == 'TSNE':
        out_desc = model.transform(input_data)
    if method == 'Isomap':
        out_desc = model.transform(input_data)
    #####深度学习的方法
    if method=='AUTO':
        out_desc = Auto_Encoder(model, input_data, pre_trained_model_type, data_type)
    return out_desc

def Auto_Encoder(model, desc, pre_trained_model_type, data_type):

    if data_type == 'cuda':

        desc = torch.from_numpy(desc).cuda()

    if data_type == 'cpu':
        
        desc = torch.from_numpy(desc)

    if pre_trained_model_type =='alexnet':
        desc = desc.view(desc.size(0),256,11,11)

    if pre_trained_model_type =='vgg16':
        desc = desc.view(desc.size(0),512,26,26)

    if pre_trained_model_type =='resnet101':
        desc = desc.view(desc.size(0),1024,12,12)

    if pre_trained_model_type == 'densenet169':
        desc = desc.view(desc.size(0),1280,12,12)
    
    encoder, decoder = model(desc)

    encoder = encoder.cpu().detach().numpy()

    return encoder
