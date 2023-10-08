# -*- coding: utf-8 -*-
# @Time    : 8/4/21 4:30 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : get_norm_stats.py

# this is a sample code of how to get normalization stats for input spectrogram

import torch
import numpy as np

import dataloader

# set skip_norm as True only when you are computing the normalization stats
audio_conf = {'num_mel_bins': 128, 'target_length': 3000, 'freqm': 0, 'timem': 0, 'mixup': 0, 'skip_norm': True, 'mode': 'train', 'dataset': 'audioset'}
mean=[]
std=[]
# for fold in range(1, 11):
#     train_loader = torch.utils.data.DataLoader(
#     dataloader.AudiosetDataset('/workspace/lh/ast-adapter/tasks/urbansound8k/data/datafiles/urbansound8k_train_data_{}.json'.format(fold), label_csv='/workspace/lh/ast-adapter/tasks/urbansound8k/data/class_labels_indices.csv',
#                                 audio_conf=audio_conf), batch_size=1000, shuffle=False, num_workers=8, pin_memory=True)
#
#     for i, (audio_input, labels) in enumerate(train_loader):
#         cur_mean = torch.mean(audio_input)
#         cur_std = torch.std(audio_input)
#         mean.append(cur_mean)
#         std.append(cur_std)
#         print(cur_mean, cur_std)
train_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset('/workspace/lh/ast-adapter/tasks/gtzan/data/datafiles/gtzan_test_data.json', label_csv='/workspace/lh/ast-adapter/tasks/gtzan/data/class_labels_indices.csv',
                                audio_conf=audio_conf), batch_size=1000, shuffle=False, num_workers=8, pin_memory=True)

for i, (audio_input, labels) in enumerate(train_loader):
    cur_mean = torch.mean(audio_input)
    cur_std = torch.std(audio_input)
    mean.append(cur_mean)
    std.append(cur_std)
    print(cur_mean, cur_std)
print(np.mean(mean), np.mean(std))