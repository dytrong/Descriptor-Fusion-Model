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
