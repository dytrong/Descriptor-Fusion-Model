import torch.nn as nn
import torchvision.models as models  
import torch
import time
import sys
import os
import configparser
import argparse
from sklearn import preprocessing

sys.path.append('../utils/')
from compute_distance import compute_cross_correlation_match
from compute_average_precision import compute_AP
from compute_keypoints_patch import * 
from forward import *
from preprocess import change_patch_size, normalization_desc
from pre_trained_desc import *

sys.path.append("../CAE/")
from dimension_reduce_method import select_dimension_reduce_model, dimension_reduce_method

######初始化参数
config = configparser.ConfigParser()
config.read('../utils/setup.ini')
Model_Img_size = config.getint("DEFAULT", "Model_Image_Size")
Max_kp_num = config.getint("DEFAULT", "Max_kp_num")
img_suffix = config.get("DEFAULT", "img_suffix")
Image_data = config.get("DATASET", "Hpatch_Image_Path")

######接收参数
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=['i','v'], required=True)
parser.add_argument("--dimension", type=int, choices=[128,256,512,1024,2048,4096], required=True)
parser.add_argument("--reduce_flag", type=int, choices=[0,1], required=True)
parser.add_argument("--reduce_method", type=str,choices=['PCA', 'RP', 'AUTO'], required=True)

parser.add_argument("--autoencoder_parameter_path", type=str, required=True)
parser.add_argument("--pre_trained_model_name", type=str, choices=['alexnet','resnet101','densenet169'])
parser.add_argument("--pre_trained_descs_path", type=str, required=False)

parser.add_argument("--fusion_flag", type=int, choices=[0,1], required=True)
parser.add_argument("--fusion_method", type=str, choices=['cat', 'sum', 'mul', 'AE'], required=False)
parser.add_argument("--fusion_dimension", type=int, choices=[128,256], required=False)
parser.add_argument("--fusion_epoch", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--lr", type=float)
args = parser.parse_args()

def compute_batch_descriptors(mynet, input_data, mini_batch_size=8):

    #####计算描述符
    desc = generate_des(mynet, input_data.cuda(), mini_batch_size, args.pre_trained_model_name).descriptor

    return desc

#######数据融合
def data_fusion(desc1, desc2, fusion_method = 'cat'):

    fusion_method = args.fusion_method

    ######拼接concatenate((array1,array2),axis=1)表示横向拼接
    if fusion_method == 'cat':

        desc = np.concatenate((desc1, desc2), axis=1)

    if fusion_method == 'sum':

        desc = desc1 + desc2

    if fusion_method == 'mul':

        desc = desc1 * desc2
     
    if fusion_method == 'AE':

        desc = np.concatenate((desc1, desc2), axis=1)

        desc = normalization_desc(desc)

        sys.path.append("../FCAE/")
            
        from train_FCAE import fusion_test_autoencoder

        AE_model_parameters_path = "../FCAE/model_parameters/"
        
        AE_model_parameters_path_1 = AE_model_parameters_path + "100_SAE_4224_densenet169_hardnet_1024_64_0.0001.pth"
        
        desc = fusion_test_autoencoder(AE_model_parameters_path_1, 4224, 1024, 'CPU', desc)

        AE_path = AE_model_parameters_path + "100_SAE_1024_densenet169_hardnet_256_64_0.0001.pth"

        desc = fusion_test_autoencoder(AE_path, 1024, 256, 'CPU', desc)

    return desc

def Hardnet_desc(img_patch):

    #####导入HardNet代码路径
    sys.path.append('/home/data1/daizhuang/pytorch/Descriptor-Fusion-Model/FCAE/hardnet/')

    from hardnet_desc import extract_hardnet_desc

    hardnet_desc = extract_hardnet_desc(img_patch)

    if args.dataset == 'i':

        hardnet_desc -= 0.0001433593

        hardnet_desc /= 0.08838826
    if args.dataset == 'v':

        hardnet_desc -= -0.000108431

        hardnet_desc /= 0.08838831

    return hardnet_desc


def compute_fusion_model_descriptors(mynet,
                                     Img_path, 
                                     reduce_model, 
                                     reduce_flag = args.reduce_flag,
                                     fusion_flag = args.fusion_flag
                                     ):

    cv_kpts, img_patch = compute_valid_keypoints(Img_path, Max_kp_num)

    densenet169_input_data = change_patch_size(img_patch, 224, 3)

    ######pre-trained CNN descriptor
    desc = compute_batch_descriptors(mynet, densenet169_input_data)

    print("计算的描述符为:" + args.pre_trained_model_name)

    print("检测出的有效特征点数量为:" + str(desc.shape[0]))

    print("降维前描述符维度为:" + str(desc.shape[1]))

    #####reduce dimension
    if reduce_flag:

        ######标准化
        if args.dataset == "i":

            start = time.time()
        
            desc = normalization_desc(desc)

            print("描述符正则化耗时:" + str(time.time()-start))

        start = time.time()

        desc = dimension_reduce_method(reduce_model, desc, args.reduce_method, args.pre_trained_model_name)

        #if args.dataset == 'i':

            #desc = normalization_desc(desc)

        print("描述符降维耗时:" + str(time.time()-start))

        print("降维后的描述符维度:" + str(desc.shape[1])) 

    if fusion_flag:

        ####HardNet descriptors
        hardnet_input_data = change_patch_size(img_patch, 32, 1)

        hardnet_desc = Hardnet_desc(hardnet_input_data.cuda())

        ###descriptor fusion
        start = time.time()

        desc = data_fusion(desc, hardnet_desc)

        print("描述符融合耗时:" + str(time.time()-start))

        print('融合后描述符维度:' + str(desc.shape[1]))
  
    return cv_kpts, desc

def compute_mAP(mynet, file_path, reduce_dimension_model):

    total_AP = []

    extract_desc_time = []

    compute_desc_dis_time = []

    total_desc = []

    #####子数据集地址
    base_path = Image_data + str(file_path) + '/'

    ####第１对图像对
    print("start compute the 1 pairs matches")

    Img_path_A = base_path + str(1) + img_suffix

    img1 = cv2.imread(Img_path_A)

    kp1, desc1 = compute_fusion_model_descriptors(mynet, Img_path_A, reduce_dimension_model)

    desc = np.zeros((1, desc1.shape[1]), dtype = 'float32')

    desc = np.vstack((desc, desc1))

    for i in range(2,7):

        print("start compute the "+str(i)+" pairs matches")

        img_save_path = '/home/zhuang/Fusions/Descriptors-Fusion/results/' + "ours_" + str(file_path) + '_' + str(i) + '.jpg'

        ####ground truth of Homography 
        H_path = base_path + 'H_1_' + str(i)

        ####读取图片
        Img_path_B = base_path + str(i) + img_suffix

        img2 = cv2.imread(Img_path_B)

        #############提取特征点，和卷积描述符
        start = time.time()

        kp2, desc2 = compute_fusion_model_descriptors(mynet, Img_path_B, reduce_dimension_model)

        desc = np.vstack((desc, desc2))

        extract_desc_time.append(time.time()-start)

        ##############计算描述符之间的距离并寻找特征点匹配对
        start = time.time()

        ##### L2, cos
        distance_method = 'cos'

        match_cc = compute_cross_correlation_match(distance_method, des1=desc1, des2=desc2)

        compute_desc_dis_time.append(time.time()-start)

        ##############compute average precision(AP)
        AP = compute_AP(img1, img2, kp1, kp2, match_cc, H_path, distance_method, img_save_path, imshow=False)

        total_AP.append(AP)

    mAP = np.mean(total_AP)

    print('提取描述符平均耗时:'+str(np.mean(extract_desc_time)))

    print('计算描述符距离平均耗时:'+str(np.mean(compute_desc_dis_time)))

    print('5幅图像的平均精度:'+str(mAP))

    return mAP, desc[1:,]


if __name__ == "__main__":

    start = time.time()

    all_mAP = []

    Count = 0 

    save_flag = False

    all_desc = []

    if save_flag:

        desc = np.zeros((1, 256), dtype = 'float32')

    #####选择预训练神经网络模型
    mynet = select_pretrained_cnn_model(flag = args.pre_trained_model_name)

    #####返回降维模型
    if args.reduce_flag:

        reduce_dimension_model = select_dimension_reduce_model(args.reduce_method, 
                                                               args.dimension,
                                                               args.pre_trained_descs_path,
                                                               args.autoencoder_parameter_path,
                                                               args.pre_trained_model_name
                                                              )
    else:

        reduce_dimension_model = None

    #####遍历图像数据集, 输出所有数据的平均精度
    for roots, dirs, files in os.walk(Image_data):

        for Dir in dirs:

            if Dir[0] == args.dataset:

                print('读取的图像:'+Dir)

                Count = Count + 1

                print('读取的图片张数:'+str(Count))

                mAP, total_desc = compute_mAP(mynet, Dir, reduce_dimension_model)

                if save_flag:

                    desc = np.vstack((desc, total_desc))

                all_mAP.append(mAP)

                print('\n')

    print('所有数据的平均精度为:'+str(np.sum(all_mAP)/len(all_mAP)))

    print('总共耗时:'+str(time.time()-start))

    if save_flag:
                                                                                                                       
        fusion_desc_path = '/home/data1/daizhuang/pytorch/Descriptor-Fusion-Model/FCAE/fusion_descriptors/densenet169/' + \
                           args.dataset + '_' + str(desc.shape[1]) + '_' + args.pre_trained_model_name + '_hardnet_all_descs.npy'

        np.save(fusion_desc_path, desc[1:,])
