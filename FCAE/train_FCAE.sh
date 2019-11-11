#! /bin/zsh
input_size=4224
output_size=256

###fusion descriptor dimension 256 + 4096, 4224
#fusion_dimension=4224
EPOCH=300
pre_trained_cnn='densenet169'
batch_size=64
lr=0.0001

####融合描述幅的位置
dataset_path="./fusion_descriptors/${pre_trained_cnn}/"

echo "融合描述符的位置: ${pre_trained_descs_path}"

while [ ${output_size} != '512' ]
do

###每隔100 epoch保存一次模型参数
autoencoder_parameter_path="${input_size}_${pre_trained_cnn}_hardnet_${output_size}_${batch_size}_${lr}.pth"

echo "模型参数保存的位置:${autoencoder_parameter_path}"

(CUDA_VISIBLE_DEVICES=1  python3 -u  train_FCAE.py \
--dataset_path ${dataset_path} \
--pre_trained_cnn ${pre_trained_cnn} \
--input_size ${input_size} \
--output_size ${output_size} \
--autoencoder_parameter_path ${autoencoder_parameter_path} \
--EPOCH ${EPOCH} \
--batch_size ${batch_size} \
--lr ${lr} \
> "./log/${EPOCH}_${input_size}_${pre_trained_cnn}_hardnet_${output_size}_${batch_size}_${lr}.txt"
)
output_size=$((output_size*2))
done
