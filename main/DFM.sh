#! /bin/zsh

EPOCH=200
dataset='i'
dimension=4096

####PCA, RP, AUTO, no_reduce
reduce_flag=1
reduce_method='AUTO'

####alexnet, resnet101, densenet169
pre_trained_model_name='densenet169'

####sum, cat, mul, AE
fusion_flag=1
fusion_method='AE'

####128, 256
fusion_dimension=128

####AE fusion train epoch
fusion_epoch=200
batch_size=64
lr=0.0001

####-u是为了禁止缓存，让结果可以直接进入日志文件
while [ ${fusion_dimension} != '256' ]
do
######auto encoder 预训练模型地址
####alexnet, resnet, densenet
autoencoder_parameter_path="/home/data1/daizhuang/pytorch/Descriptor-Fusion-Model/CAE/parameters/${EPOCH}_${pre_trained_model_name}_${dimension}_autoencoder_cnn.pth"

echo "Auto-Encoder降维模型参数地址:${autoencoder_parameter_path}"

#####训练数据集地址(PCA, RP), 主有使用PCA和RP时,才需要pre-trained-descs-path
pre_trained_descs_path="./CAE/parameters/descriptors/${pre_trained_model_name}/${dataset}_${model_name}_all_descs.npy"

(CUDA_VISIBLE_DEVICES=0 python3 -u  DFM.py \
--dataset ${dataset} \
--dimension ${dimension} \
--reduce_flag ${reduce_flag} \
--reduce_method ${reduce_method} \
--autoencoder_parameter_path ${autoencoder_parameter_path} \
--pre_trained_model_name ${pre_trained_model_name} \
--pre_trained_descs_path ${pre_trained_descs_path} \
--fusion_flag ${fusion_flag} \
--fusion_method ${fusion_method} \
--fusion_dimension ${fusion_dimension} \
--fusion_epoch ${fusion_epoch} \
--batch_size ${batch_size} \
--lr ${lr} \
> "./log/${EPOCH}_train_test_${dataset}_${dimension}_${pre_trained_model_name}_${reduce_method}_${fusion_method}_${fusion_dimension}_${fusion_epoch}_${batch_size}_${lr}.log"
)
fusion_dimension=$((fusion_dimension*2))
done
