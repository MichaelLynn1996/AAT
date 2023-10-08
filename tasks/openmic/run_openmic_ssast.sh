#!/bin/bash

set -x
export TORCH_HOME=../../pretrained_models
export CUDA_VISIBLE_DEVICES=0

model=ssast
dataset=openmic
imagenetpretrain=True
audiosetpretrain=True
bal=none
lr=1e-5
epoch=30
freqm=48
timem=192
mixup=0.5
batch_size=12
fstride=10
tstride=10

dataset_mean=-4.0147066
dataset_std=4.0522056
audio_length=1024
noise=True

metrics=mAP
loss=BCE
warmup=True
lrscheduler_start=10
lrscheduler_step=5
lrscheduler_decay=0.5

warmup=True
wa=True

tr_data=./data/datafiles/openmic_train_data.json
val_data=./data/datafiles/openmic_test_data.json
exp_dir=./exp/${model}-${dataset}-f$fstride-t$tstride-p$imagenetpretrain-b$batch_size-lr${lr}-joint

if [ -d $exp_dir ]; then
  echo 'exp exist'
  exit
fi
mkdir -p $exp_dir

CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${val_data} --exp-dir $exp_dir \
--label-csv ./data/class_labels_indices.csv --n_class 20 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length} --noise ${noise} \
--metrics ${metrics} --loss ${loss} --warmup ${warmup} --lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
--tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain --audioset_pretrain $audiosetpretrain --finetune head mlp spatial prompt > $exp_dir/log.txt
