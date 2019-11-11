import torch
import numpy as np
from sklearn import preprocessing

##################compute distance matrix#############
######################################################
######euclidean distance
######Input tensor1(m,n),tensor2(k,n)
######Output tensor matrix(m,k)
def euclidean_distance(des1,des2):
    pdist = torch.nn.PairwiseDistance(2)
    dis_matrix = torch.zeros((len(des1),len(des2))).cuda()
    for i in range(len(des1)):
        dis_matrix[i] = pdist(des1[i].reshape(1,-1),des2)
    return dis_matrix.cpu().numpy()

def euclidean_distance_numpy(des1, des2):
    dis_matrix = np.zeros((len(des1),len(des2)))
    for i in range(len(des1)):
        dis_matrix[i] = np.linalg.norm(des1[i].reshape(1,-1) - des2, axis=1) 
    return dis_matrix

#####余弦距离等价于欧氏距离中向量L2-normalization的结果
#####cosine_dis=(eucl_dis^2)/(-2)+1
def cosine_distance_1(des1,des2):
    des1=preprocessing.normalize(des1,norm='l2')
    des2=preprocessing.normalize(des2,norm='l2')
    des1=torch.from_numpy(des1).cuda()
    des2=torch.from_numpy(des2).cuda()
    eucl_dis=euclidean_distance(des1,des2)
    eucl_dis_T=torch.transpose(eucl_dis,0,1)
    #####对应元素相乘
    cosine_dis=torch.mul(eucl_dis,eucl_dis)/(-2)+1
    return cosine_dis.cpu().numpy()

#######利用余弦距离公式求得
def cosine_distance_2(des1,des2):
    #### 转置
    des1_T=torch.transpose(des1,0,1)
    des2_T=torch.transpose(des2,0,1)
    #### torch.mm 矩阵的点乘 ，data.cpu().numpy() 将gpu的tensor 转化为cpu的numpy
    temp_1=torch.mm(des1,des2_T).cpu().numpy()
    temp_2=torch.pow(torch.mm(des1,des1_T),0.5).cpu().numpy()
    temp_3=torch.pow(torch.mm(des2,des2_T),0.5).cpu().numpy()
    #####初始化矩阵
    temp_matrix=np.zeros((temp_2.shape[0],temp_3.shape[0]))
    for i in range(temp_2.shape[0]):
        for j in range(temp_3.shape[0]):
            ####取对角线元素相乘
            temp_matrix[i,j]=temp_2[i,i]*temp_3[j,j]
    #####cosine distance
    cos_dis_matrix=temp_1/temp_matrix
    return cos_dis_matrix

###Compute distance matrix, including euclidean and cosine distance#######
###Input des1 is a numpy.array####
###Input des2 is a numpy.array####
def compute_dis_matrix(des1,des2,method):
    des1=torch.from_numpy(des1).cuda()
    des2=torch.from_numpy(des2).cuda()
    if method=='L2':
        dis_matrix=euclidean_distance(des1,des2)
    if method=='cos':
        dis_matrix=cosine_distance_2(des1,des2)   
    return dis_matrix

#############################compute matches#################
#############################################################
def compute_nearest_neighbor_match(method, dis_matrix=None, des1=None, des2=None):
    ####compute distance matrix
    if dis_matrix is None:
        dis_matrix = compute_dis_matrix(des1,des2,method)
    ####matches data structure
    matches = np.dtype({'names':['i','j','MAX'],'formats':['i','i','f']})
    #####init matches matrix
    Matches_left = np.zeros(dis_matrix.shape[0],dtype=matches)
    #####compute nearest neighbor match from left to right
    for i in range(dis_matrix.shape[0]):
        if method == 'L2':
            MIN_L2_DIS = np.min(dis_matrix[i,:])
            Matches_left[i]['i'] = i
            Matches_left[i]['j'] = np.argmin(dis_matrix[i,:])
            Matches_left[i]['MAX'] = MIN_L2_DIS
        if method == 'cos':
            MAX_COS_DIS = np.max(dis_matrix[i,:])
            Matches_left[i]['i'] = i
            Matches_left[i]['j'] = np.argmax(dis_matrix[i,:])
            Matches_left[i]['MAX'] = MAX_COS_DIS    
    return (Matches_left,dis_matrix) 

def compute_cross_correlation_match(method, dis_matrix=None, des1=None, des2=None):
    Matches_left,dis_matrix = compute_nearest_neighbor_match(method, dis_matrix, des1, des2) 
    Matches_right = np.zeros(dis_matrix.shape[1], dtype=Matches_left.dtype)
    ####compute match from right to left
    for j in range(dis_matrix.shape[1]):
        if method == 'L2':
            MIN_L2_DIS = np.min(dis_matrix[:,j])
            Matches_right[j]['MAX'] = MIN_L2_DIS
            Matches_right[j]['i'] = j
            Matches_right[j]['j'] = np.argmin(dis_matrix[:,j])
        if method == 'cos':
            MAX_COS_DIS = np.max(dis_matrix[:,j])
            Matches_right[j]['MAX'] = MAX_COS_DIS
            Matches_right[j]['i'] = j
            Matches_right[j]['j'] = np.argmax(dis_matrix[:,j])
    Match_CC = []
    for i in range(dis_matrix.shape[0]):
        if Matches_right[Matches_left[i]['j']]['j']==i:
            Match_CC.append([Matches_left[i]['i'], Matches_left[i]['j'], Matches_left[i]['MAX']])
    Match_CC=np.array(Match_CC)
    return Match_CC
