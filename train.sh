#!/bin/sh

train_list_path="./train.lst"
test_list_path="./test.lst"

output_dir="./output/new_v1"
checkpoint=""

learning_rate=0.1
warm_up_step=5000
display_interval=100

epochs=300
batch_size=96
img_height=448
img_width=448
use_ori=0
use_transform=0
debug=0
model_name="restormer"

data_param="--img_height=$img_height  --img_width=$img_width --use_ori=$use_ori --train_list_path=$train_list_path --test_list_path=$test_list_path"
train_param="--epochs=$epochs --learning_rate=$learning_rate --warm_up_step=$warm_up_step --batch_size=$batch_size --display_interval=$display_interval --debug=$debug --use_transform=$use_transform --num_gpus=8"
model_param="--checkpoint=$checkpoint --output_dir=$output_dir --model_name=$model_name"

# export CUDA_VISIBLE_DEVICES=0
rm -rf $output_dir
mkdir -p $output_dir
echo "python3 -m tal.img.train $data_param $train_param $model_param"
python -m tal.img.train $data_param $train_param $model_param



