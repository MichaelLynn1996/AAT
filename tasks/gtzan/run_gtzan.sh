#!/bin/bash
#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2]
##SBATCH -p sm
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=48000
#SBATCH --job-name="ast-sc"
#SBATCH --output=./log_%j.txt

set -x
# comment this line if not running on sls cluster
#. /data/sls/scratch/share-201907/slstoolchainrc
#source ../../venvast/bin/activate
export TORCH_HOME=../../pretrained_models
export CUDA_VISIBLE_DEVICES=0,1

model=ssast
dataset=gtzan
imagenetpretrain=True
audiosetpretrain=True
bal=none
lr=1e-5
epoch=30
freqm=48
timem=576
batch_size=2
fstride=10
tstride=10

dataset_mean=-3.0708861
dataset_std=3.4615202
audio_length=3000
noise=False

metrics=acc
loss=CE
warmup=False
mixup=0
lrscheduler_start=5
lrscheduler_step=1
lrscheduler_decay=0.85

s_adapter=spatial
mlp_adapter=mlp
b_adapter=batch

tr_data=./data/datafiles/gtzan_train_data.json
val_data=./data/datafiles/gtzan_val_data.json
eval_data=./data/datafiles/gtzan_test_data.json
exp_dir=./exp/${model}-${dataset}-f$fstride-t$tstride-p$imagenetpretrain-b$batch_size-lr${lr}-prompt

if [ -d $exp_dir ]; then
  echo 'exp exist'
  exit
fi
mkdir -p $exp_dir

#python ./prep_sc.py

CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${val_data} --data-eval ${eval_data} --exp-dir $exp_dir \
--label-csv ./data/class_labels_indices.csv --n_class 10 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length} --noise ${noise} \
--metrics ${metrics} --loss ${loss} --warmup ${warmup} --lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
--tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain --audioset_pretrain $audiosetpretrain --finetune head prompt > $exp_dir/log.txt
