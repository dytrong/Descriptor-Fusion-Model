import torch 
from torch import nn 
import torchvision 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
import numpy as np
import time
import argparse
from autoencoder_models import *

###reproducible
torch.manual_seed(1)

torch.cuda.manual_seed(1)

def normalization(input_data):
    mean_data = input_data.mean(axis=1, keepdims=True)
    std_data = input_data.std(axis=1, keepdims=True)
    print('Sample mean:'+str(mean_data))
    print('Sample variance:'+str(std_data))
    output_data = (input_data - mean_data) / std_data
    return output_data

def regularization(input_data):
    MAX_NUM = np.max(input_data)
    MIN_NUM = np.min(input_data)
    print('Max value:'+str(MAX_NUM))
    print('Min value:'+str(MIN_NUM))
    output_data = (input_data - MIN_NUM) / float(MAX_NUM - MIN_NUM)
    output_data = (output_data - 0.5) / 0.5
    return output_data

def random_select_samples(train_data, sample_number):
    row_rand_array = np.arange(train_data.shape[0])
    np.random.shuffle(row_rand_array)
    train_data = train_data[row_rand_array[0:sample_number]]
    train_data = train_data.astype('float32')
    return train_data

def change_data_to_fit_model(train_data, dataset_type):
    train_data = torch.from_numpy(train_data).float()
    ######alexnet dataset
    if dataset_type == 'alexnet':
        train_data = train_data.view(train_data.size(0), 256, 11, 11)
    ######vgg16 dataset
    if dataset_type == 'vgg16':
        train_data = train_data.view(train_data.size(0), 512, 26, 26)
    ######resnet101 dataset
    if dataset_type == 'resnet101':
        train_data = train_data.view(train_data.size(0), 1024, 12, 12)
    ######Densenet169 dataset
    if dataset_type == 'densenet169':
        train_data = train_data.view(train_data.size(0), 1280, 12, 12)
    print('training dataset size:'+str(train_data.shape))
    return train_data

def load_trained_data(dataset_path_ill,dataset_path_view, dataset_type = 'densenet169'):
    start = time.time()
    all_data_ill = np.load(dataset_path_ill)
    #all_data_ill = np.delete(all_data_ill, range(0, len(all_data_ill), 2), 0)
    all_data_ill = all_data_ill[::8, :]
    all_data_view = np.load(dataset_path_view)
    #all_data_view = np.delete(all_data_view, range(0, len(all_data_view), 2), 0)
    all_data_view = all_data_view[::8, :]
    print("load dataset cost time:"+str(time.time()-start))
    all_data = np.vstack((all_data_ill, all_data_view))
    all_data = normalization(all_data)
    all_data = change_data_to_fit_model(all_data, dataset_type)
    return all_data

def select_pretrained_model(model_type, dimension):
    if model_type == 'alexnet':
        AutoEncoder = AutoEncoder_alexnet(dimension).cuda()
    if model_type == 'vgg16':
        AutoEncoder = AutoEncoder_vgg16(dimension).cuda()
    if model_type == 'resnet101':
        AutoEncoder = AutoEncoder_resnet101(dimension).cuda()
    if model_type == 'densenet169':
        AutoEncoder = AutoEncoder_densenet169(dimension).cuda()
    return AutoEncoder

#####load model
def autoencoder(model_parameters_path, model_type, dimension, pretrained=False):
    model = select_pretrained_model(model_type, dimension)
    if pretrained:
        model.load_state_dict(torch.load(model_parameters_path))
    return model
