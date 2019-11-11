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

sys.path.append('/home/data1/daizhuang/pytorch/Descriptor-Fusion-Model/FCAE/')
from autoencoder_model import  autoencoder_prelu_4224

###reproducible
torch.manual_seed(1)
torch.cuda.manual_seed(1)

def normalization(input_data):

    mean_data = input_data.mean(axis=1, keepdims = True)

    std_data = input_data.std(axis=1, keepdims = True)

    print('样本均值为:'+str(mean_data))

    print('样本方差为:'+str(std_data))

    output_data = (input_data - mean_data) / std_data

    return output_data


def change_arr_to_tensor(input_data):

    output_data = torch.from_numpy(input_data)

    output_data = output_data.float()

    return output_data

###下载训练数据集,对57张图像中所有特征点进行训练
def load_trained_data(dataset_path_ill, dataset_path_view):

    all_data_ill = np.load(dataset_path_ill)

    all_data_view = np.load(dataset_path_view)

    all_data = np.vstack((all_data_ill, all_data_view))

    all_data = normalization(all_data)

    train_data = np.delete(all_data, range(0, len(all_data), 500), 0)
 
    test_data = all_data[::500, :]
    
    train_data = change_arr_to_tensor(train_data)

    test_data = change_arr_to_tensor(test_data)

    print('训练autoencoder训练数据集大小为:'+str(train_data.shape))

    print('训练autoencoder测试数据集大小为:'+str(test_data.shape))

    return train_data, test_data

def fusion_autoencoder(model_parameters_path, input_size, output_size, flag='GPU', **kwargs):
    
    model = autoencoder_prelu_4224(dimension)

    ####加载cpu或者gpu模型
    if flag == 'GPU':

        model.load_state_dict(torch.load(model_parameters_path))

    if flag == 'CPU':

        model.load_state_dict(torch.load(model_parameters_path, map_location=lambda storage, loc: storage))

    return model

def fusion_test_autoencoder(model_parameters_path, input_size, output_size, flag, desc):

    model = fusion_autoencoder(model_parameters_path, 
                               input_size, 
                               output_size, 
                               flag)
    model.eval()

    desc = torch.from_numpy(desc)

    encoder, decoder = model(desc)

    encoder = encoder.cpu().detach().numpy()

    return encoder

def train(model, train_dataloader, test_dataloader, optimizer, loss_func, model_parameters_path, EPOCH):
    
    for epoch in range(EPOCH):
 
        model.train()

        train_loss = 0

        train_step = 0

        for train_mini_data in train_dataloader:

            #########forward#########
            optimizer.zero_grad()

            encoded, decoded = model(train_mini_data.cuda())

            loss = loss_func(decoded, train_mini_data.cuda())

            #########backward########
            loss.backward()

            #######record loss value#####
            train_loss = train_loss + loss.item()

            optimizer.step()

            train_step += 1

        ##########eval################################
        model.eval()

        test_loss = 0

        test_step = 0

        for test_mini_data in test_dataloader:

            test_encoded, test_decoded = model(test_mini_data.cuda())

            loss = loss_func(test_decoded, test_mini_data.cuda())

            test_loss = test_loss + loss.item()

            test_step += 1

        if (epoch+1) % 5 ==  0:

            print('epoch [{}/{}], train loss:{:.4f}, \teval loss:{:.4f}'.format(epoch+1, EPOCH, train_loss/train_step, test_loss/test_step))
        
        if (epoch+1) % 100 == 0:

            ######保存模型参数
            model_parameters = str(epoch+1) + '_' + model_parameters_path
 
            model_para_path = './model_parameters/' + model_parameters

            torch.save(model.state_dict(), model_para_path) 

if __name__=="__main__":
    ######接收参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--pre_trained_cnn", type=str, choices=['alexnet','resnet101', 'densenet169'], required=True)
    parser.add_argument("--input_size", type=int, required=True)
    parser.add_argument("--output_size", type=int, required=True)
    parser.add_argument("--autoencoder_parameter_path", type=str, required=True)
    parser.add_argument("--EPOCH", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    args = parser.parse_args()
    
    #####data and model path
    dataset_path = args.dataset_path
    dataset_path_ill = dataset_path + "i_" + str(args.input_size) + "_densenet169_hardnet_all_descs.npy"
    dataset_path_view = dataset_path + "v_" + str(args.input_size) + "_densenet169_hardnet_all_descs.npy"

    model_parameters_path = args.autoencoder_parameter_path

    #####download dataset
    train_data, test_data = load_trained_data(dataset_path_ill, dataset_path_view)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, num_workers=8, shuffle=True)

    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, num_workers=8, shuffle=True)

    start=time.time()

    ####upload model
    model = autoencoder_prelu_4224(args.output_size).cuda()

    ####select optimizer method
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    ####computer loss function
    loss_func = nn.MSELoss()

    ####train
    train(model, train_dataloader, test_dataloader, optimizer, loss_func, model_parameters_path, args.EPOCH)
    
    end=time.time()

    print('训练autoencoder模型参数共耗时:'+str(end-start))    
