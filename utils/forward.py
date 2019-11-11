import torch.nn.functional as F
import numpy as np
import torch
import matplotlib.pyplot as plt

#######data的形状类似512*13*13#########
def vis_square(data):
        data=data.cpu().detach().numpy()
        data = (data-data.min())/(data.max()-data.min())
        print(data.shape)
        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = (((0, n**2-data.shape[0]), (0, 1), (0, 1))+((0, 0), )*(data.ndim-3))
        data = np.pad(data, padding, mode='constant', constant_values=1)
        data = data.reshape((n, n)+data.shape[1:]).transpose((0, 2, 1, 3)+tuple(range(4, data.ndim+1)))
        data = data.reshape((n*data.shape[1], n*data.shape[3])+data.shape[4:])
        plt.imshow(data)
        plt.axis('off')
        plt.show()


def alexnet(net,input_data):
    x=net.features[0](input_data) ###conv1
    x=net.features[1](x)  ###Relu
    x=net.features[2](x)  ###max_pool
    x=net.features[3](x)  ###conv2
    x=net.features[4](x) ###Relu
    x=net.features[5](x)  ###max_pool
    x=net.features[6](x)  ###conv3
    x=net.features[7](x)  ###relu
    x=net.features[8](x)  ###conv4
    x=net.features[9](x) ###relu
    #x=net.features[10](x) ###conv5
    #x=net.features[11](x) ###relu
    #x=net.features[12](x) ###max_pool
    x=F.max_pool2d(x, kernel_size=3, stride=1, padding=0)
    #print('x的形状：'+str(x.shape))
    x=x.view(x.shape[0],-1) 
    return x


def vgg16(net,input_data):
    x=net.features[0](input_data) ###conv1_1
    x=net.features[1](x)  ###Relu
    x=net.features[2](x)  ###conv1_2
    x=net.features[3](x)  ###Relu
    x=net.features[4](x) ###maxpool1
    x=net.features[5](x)  ###conv2_1
    x=net.features[6](x)  ###Relu
    x=net.features[7](x)  ###conv2_2
    x=net.features[8](x)  ###relu
    x=net.features[9](x) ###maxpool2
    x=net.features[10](x) ###conv3_1
    x=net.features[11](x) ###Relu
    x=net.features[12](x) ###conv3_2
    x=net.features[13](x) ###Relu
    x=net.features[14](x) ###con3_3
    x=net.features[15](x) ###Relu
    x=net.features[16](x) ###maxpool3
    x=net.features[17](x) ###con4_1
    x=net.features[18](x) ###Relu
    x=net.features[19](x) ###conv4_2
    x=net.features[20](x) ###Relu
    x=net.features[21](x) ###conv4_3
    x=net.features[22](x) ###Relu
    #x=net.features[23](x) ###maxpool4
    #x=net.features[24](x) ###conv5_1,提取此处的卷积特征,效果最好
    #x=net.features[25](x) ###Relu
    #x=net.features[26](x) ###conv5_2
    #x=net.features[27](x) ###Relu
    #x=net.features[28](x) ###conv5_3
    #x=net.features[29](x) ###Relu
    #x=net.features[30](x) ###maxpool5  
    x=F.max_pool2d(x, kernel_size=3, stride=1, padding=0)
    #print('x的形状：'+str(x.shape))
    x=x.view(x.shape[0],-1)
    #x = net.classifier(x)
    return x   

def vgg19(net,input_data):
    x=net.features[0](input_data) ###conv1_1
    x=net.features[1](x)  ###Relu
    x=net.features[2](x)  ###conv1_2
    x=net.features[3](x)  ###Relu
    x=net.features[4](x) ###maxpool1
    x=net.features[5](x)  ###conv2_1
    x=net.features[6](x)  ###Relu
    x=net.features[7](x)  ###conv2_2
    x=net.features[8](x)  ###relu
    x=net.features[9](x) ###maxpool2
    x=net.features[10](x) ###conv3_1
    x=net.features[11](x) ###Relu
    x=net.features[12](x) ###conv3_2
    x=net.features[13](x) ###Relu
    x=net.features[14](x) ###con3_3
    x=net.features[15](x) ###Relu
    x=net.features[16](x) ###con3_4
    x=net.features[17](x) ###relu
    x=net.features[18](x) ###maxpool3
    #x=net.features[19](x) ###conv4_1
    #x=net.features[20](x) ###Relu
    #x=net.features[21](x) ###conv4_2
    #x=net.features[22](x) ###Relu
    #x=net.features[23](x) ###conv4_3
    #x=net.features[24](x) ###relu
    #x=net.features[25](x) ###conv4_4
    #x=net.features[26](x) ###relu
    #x=net.features[27](x) ###maxpool4
    #x=net.features[28](x) ###conv5_1
    #x=net.features[29](x) ###Relu
    #x=net.features[30](x) ###conv5_2  
    #x=net.features[31](x) ###relu
    #x=net.features[32](x) ###conv5_3
    #x=net.features[33](x) ###relu
    #x=net.features[34](x) ####conv5_4
    #x=net.features[35](x) ####relu
    #x=net.features[36](x) ####maxpool5
    x=x.view(x.shape[0],-1)
    #x=net.classifier(x)
    return x

def inception_v3(net,input_data):
    x=net.Conv2d_1a_3x3(input_data)
    x=net.Conv2d_2a_3x3(x)
    x=net.Conv2d_2b_3x3(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    x = net.Conv2d_3b_1x1(x)
    x = net.Conv2d_4a_3x3(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    x = net.Mixed_5b(x)
    #x = net.Mixed_5c(x)
    #x = net.Mixed_5d(x)
    #x = net.Mixed_6a(x)
    #x = net.Mixed_6b(x)
    #x = net.Mixed_6c(x)
    #x = net.Mixed_6d(x)  #####提取此处的卷积特征,效果最好
    #x = net.Mixed_6e(x)
    #x = net.AuxLogits(x)
    #x = net.Mixed_7a(x)
    #x = net.Mixed_7b(x)
    #x = net.Mixed_7c(x)
    x=x.view(x.shape[0],-1)
    return x

def squeezenet1(net,input_data):
    x=net.features[0](input_data) ###conv25
    x=net.features[1](x)  ###Relu
    x=net.features[2](x)  ###maxpool2d
    x=net.features[3](x)  ###Fire ,squeezenet的一个块
    x=net.features[4](x)  ###Fire
    x=net.features[5](x)  ###Fire
    x=net.features[6](x)  ###maxpool2d
    x=net.features[7](x)  ###Fire
    x=net.features[8](x)  ###Fire
    x=net.features[9](x)  ###Fire
    x=net.features[10](x) ###Fire
    x=net.features[11](x) ###maxpool2d
    x=net.features[12](x) ###Fire
    x=x.view(x.shape[0],-1)
    #x = net.classifier(x)
    return x

def resnet(net,input_data):
    x=net.conv1(input_data)
    x=net.bn1(x)
    x=net.relu(x)
    x=net.maxpool(x)
    x=net.layer1(x)
    x=net.layer2(x)
    x=net.layer3(x)
    #x=net.layer4(x)
    #x=net.avgpool(x)
    #x=F.avg_pool2d(x, kernel_size=3, stride=1, padding=0)
    x=F.max_pool2d(x, kernel_size=3, stride=1, padding=0)
    #print('x的形状：'+str(x.shape))
    x=x.view(x.shape[0],-1)
    return x

def densenet(net,input_data):
    x=net.features[0](input_data) ###conv1
    x=net.features[1](x)  ###batch_norm1
    x=net.features[2](x)  ###relu
    x=net.features[3](x) ###maxpool1
    x=net.features[4](x) #_DenseBlock 1
    x=net.features[5](x) #_Transition 1
    x=net.features[6](x) #_DenseBlock 2
    x=net.features[7](x) #_Transition 2
    x=net.features[8](x) #_DenseBlock 3
    #x=net.features[9](x) #_Transition 3
    #x=net.features[10](x) #_DenseBlock 4
    #x=net.features[11](x) # batch_norm
    x=F.max_pool2d(x, kernel_size=3, stride=1, padding=0)
    #print('x的形状：'+str(x.shape))
    x=x.view(x.shape[0], -1)
    return x
