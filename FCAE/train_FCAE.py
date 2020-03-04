import torch 
from torch import nn 
import torchvision 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
import numpy as np
import time
import argparse
import os
import sys

sys.path.append('../FCAE/')
from autoencoder_model import autoencoder_prelu_4224

###reproducible
torch.manual_seed(1)
torch.cuda.manual_seed(1)

def fusion_autoencoder(model_parameters_path, output_dim, flag='GPU', **kwargs):
    
    model = autoencoder_prelu_4224(output_dim)

    ####加载cpu或者gpu模型
    if flag == 'GPU':

        model.load_state_dict(torch.load(model_parameters_path))

    if flag == 'CPU':

        model.load_state_dict(torch.load(model_parameters_path, map_location=lambda storage, loc: storage))

    return model

def fusion_test_autoencoder(model_parameters_path, output_dim, flag, desc):

    model = fusion_autoencoder(model_parameters_path, output_dim, flag)
    
    model.eval()

    desc = torch.from_numpy(desc)

    encoder, decoder = model(desc)

    encoder = encoder.cpu().detach().numpy()

    return encoder
