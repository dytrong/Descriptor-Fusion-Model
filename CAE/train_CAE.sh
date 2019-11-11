#! /bin/zsh
dataset='v'
dimension=4096
model_type='densenet169'
EPOCH=500

####训练数据集地址
dataset_path="/home/data1/daizhuang/pytorch/Descriptor-Fusion-Model/CAE/parameters/descriptors/${dataset}_184320_${model_type}_all_descs.npy"

while [ ${dimension} != '8192' ]
do
###Auto-Encoder模型参数地址
model_parameters_path="${model_type}_${dimension}_autoencoder_cnn.pth"
(CUDA_VISIBLE_DEVICES=1  python -u  train_CAE.py \
--dataset ${dataset} \
--dimension ${dimension} \
--model_type ${model_type} \
--EPOCH ${EPOCH} \
--dataset_path ${dataset_path} \
--model_parameters_path ${model_parameters_path} > ./log/${EPOCH}_${dataset}_${model_type}_${dimension}.txt
)
dimension=$((dimension*2))
done
