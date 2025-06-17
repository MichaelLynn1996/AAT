import numpy as np
import json
import os
from tqdm import tqdm
import pandas as pd

root = "/workspace/datasets/openmic-2018"

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]


def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]


with open(root + "/partitions/split01_train.csv", 'r') as f:
    train_list = [line.strip() for line in f.readlines()]
    print(train_list[0:5])

with open(root + "/partitions/split01_test.csv", 'r') as f:
    test_list = [line.strip() for line in f.readlines()]
    print(test_list[0:5])

aggregated_labels = pd.read_csv(root + "/openmic-2018-aggregated-labels.csv")
# class_map = json.load(root + "/class-map.json")
with open(root + "/class-map.json", 'r') as f:
    class_map = json.load(f)
new_class_map = {}
index = []
mid = []
display_name = []
for k, v in class_map.items():
    index.append(v)
    mid.append("/m/opmc"+ str(v).zfill(2))
    display_name.append(k)
    new_class_map[k] = "/m/opmc"+ str(v).zfill(2)
class_labels_indices = pd.DataFrame({"index":index, "mid":mid, "display_name":display_name})
# class_labels_indices.reset_index(drop=True, inplace=True)
if not os.path.exists('../urbansound8k/data'):
    os.mkdir('../urbansound8k/data')
class_labels_indices.to_csv('data/class_labels_indices.csv', index=False)
print(class_labels_indices)
print(new_class_map)

if not os.path.exists(root + '/audio_16k/'):

    os.mkdir(root + '/audio_16k/')

    dir_list = get_immediate_subdirectories(root + '/audio/')

    for d in dir_list:
        os.mkdir(root + '/audio_16k/' + d)
        file_list = get_immediate_files(root + '/audio/' + d)
        for audio in file_list:
            wav_path = root + '/audio_16k/' + d + '/' + audio
            command = 'sox ' + root + '/audio/' + d + '/' + audio + ' -r 16000 -c 1 ' + wav_path
            print(command)
            os.system(command)

if os.path.exists(root + '/audio_16k/'):
    train_json = []
    test_json = []
    dir_list = get_immediate_subdirectories(root + '/audio_16k/')

    for d in dir_list:
        file_list = get_immediate_files(root + '/audio/' + d)
        for audio in file_list:
            wav_path = root + '/audio_16k/' + d + '/' + audio
            query = aggregated_labels[aggregated_labels["sample_key"] == audio[:-4]]
            label_list = query.loc[:, "instrument"].tolist()
            labels = new_class_map[label_list[0]]
            for i in range(1, len(label_list)):
                labels += ","
                labels += new_class_map[label_list[i]]
            cur_dict = {"wav": wav_path, "labels": labels}
            # print(audio[:-4])
            if audio[:-4] in train_list:
                train_json.append(cur_dict)
                print(cur_dict)
            elif audio[:-4] in test_list:
                test_json.append(cur_dict)
                print(cur_dict)
            else:
                print('Invalid', audio[:-4])
    if not os.path.exists('./data/datafiles'):
        os.mkdir('./data/datafiles')
    with open('./data/datafiles/openmic_train_data.json', 'w') as f:
        json.dump({'data': train_json}, f, indent=1)
    with open('./data/datafiles/openmic_test_data.json', 'w') as f:
        json.dump({'data': test_json}, f, indent=1)

