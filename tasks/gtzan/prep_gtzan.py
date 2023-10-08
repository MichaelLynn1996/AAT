import numpy as np
import json
import os
from tqdm import tqdm
import pandas as pd
from filters import filtered_test, filtered_train, filtered_valid

root = "/workspace/datasets/GTZAN"

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]


def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]


if not os.path.exists(root + '/audio_16k/'):
    # convert the audio to 16kHz
    # base_dir = 'data/ESC-50-master/'
    os.mkdir(root + '/audio_16k/')
    dir_list = get_immediate_subdirectories(root + '/genres_original/')

    for d in dir_list:
        os.mkdir(root + '/audio_16k/' + d)
        file_list = get_immediate_files(root + '/genres_original/' + d)
        for audio in file_list:
            wav_path = root + '/audio_16k/' + d + '/' + audio
            command = 'sox ' + root + '/genres_original/' + d + '/' + audio + ' -r 16000 -c 1 ' + wav_path
            print(command)
            os.system(command)


label_set = np.loadtxt('./data/class_labels_indices.csv', delimiter=',', dtype='str')
label_map = {}
for i in range(1, len(label_set)):
    label_map[label_set[i][2].strip('"')] = label_set[i][1]
print(label_map)

if not os.path.exists('./data/datafiles'):
    os.mkdir('./data/datafiles')

train_list = []
for ft in filtered_train:
    fs = ft.split('.')
    wav_path = root + '/audio_16k/' + fs[0] + '/' + ft + '.wav'
    labels = label_map[fs[0]]
    cur_dict = {"wav": wav_path, "labels": labels}
    print(cur_dict)
    train_list.append(cur_dict)
    with open('./data/datafiles/gtzan_train_data.json', 'w') as f:
        json.dump({'data': train_list}, f, indent=1)

val_list = []
for ft in filtered_valid:
    fs = ft.split('.')
    wav_path = root + '/audio_16k/' + fs[0] + '/' + ft + '.wav'
    labels = label_map[fs[0]]
    cur_dict = {"wav": wav_path, "labels": labels}
    print(cur_dict)
    val_list.append(cur_dict)
    with open('./data/datafiles/gtzan_val_data.json', 'w') as f:
        json.dump({'data': val_list}, f, indent=1)

test_list = []
for ft in filtered_test:
    fs = ft.split('.')
    wav_path = root + '/audio_16k/' + fs[0] + '/' + ft + '.wav'
    labels = label_map[fs[0]]
    cur_dict = {"wav": wav_path, "labels": labels}
    print(cur_dict)
    test_list.append(cur_dict)
    with open('./data/datafiles/gtzan_test_data.json', 'w') as f:
        json.dump({'data': test_list}, f, indent=1)