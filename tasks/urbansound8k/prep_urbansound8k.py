import numpy as np
import json
import os

root = "/workspace/datasets/UrbanSound8K"

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]

# downlooad esc50
# dataset provided in https://github.com/karolpiczak/ESC-50
if not os.path.exists(root + '/audio_16k/'):
    # convert the audio to 16kHz
    # base_dir = 'data/ESC-50-master/'
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

label_set = np.loadtxt('./data/class_labels_indices.csv', delimiter=',', dtype='str')
label_map = {}
for i in range(1, len(label_set)):
    label_map[label_set[i][0]] = label_set[i][1]
print(label_map)

# fix bug: generate an empty directory to save json files
if os.path.exists('data/datafiles') == False:
    os.mkdir('data/datafiles')

for fold in [1,2,3,4,5,6,7,8,9,10]:
    base_path = root + "/audio_16k/"
    # meta = np.loadtxt('./data/ESC-50-master/meta/esc50.csv', delimiter=',', dtype='str', skiprows=1)
    train_wav_list = []
    eval_wav_list = []
    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        file_list = get_immediate_files(base_path + 'fold{}'.format(i))
        for file in file_list:
            meta = file.split('-')
            # cur_label = label_map[meta[1]]
            cur_dict = {"wav": base_path + 'fold{}'.format(i) + '/' + file, "labels": label_map[meta[1]]}
            print(cur_dict)
            if i == fold:
                eval_wav_list.append(cur_dict)
            else:
                train_wav_list.append(cur_dict)


    with open('./data/datafiles/urbansound8k_train_data_'+ str(fold) +'.json', 'w') as f:
        json.dump({'data': train_wav_list}, f, indent=1)

    with open('./data/datafiles/urbansound8k_eval_data_'+ str(fold) +'.json', 'w') as f:
        json.dump({'data': eval_wav_list}, f, indent=1)

    # for i in range(0, len(meta)):
    #     cur_label = label_map[meta[i][3]]
    #     cur_path = meta[i][0]
    #     cur_fold = int(meta[i][1])
    #     # /m/07rwj is just a dummy prefix
    #     cur_dict = {"wav": base_path + cur_path, "labels": '/m/07rwj'+cur_label.zfill(2)}
    #     if cur_fold == fold:
    #         eval_wav_list.append(cur_dict)
    #     else:
    #         train_wav_list.append(cur_dict)
    #
    # print('fold {:d}: {:d} training samples, {:d} test samples'.format(fold, len(train_wav_list), len(eval_wav_list)))
    #
    # with open('./data/datafiles/esc_train_data_'+ str(fold) +'.json', 'w') as f:
    #     json.dump({'data': train_wav_list}, f, indent=1)
    #
    # with open('./data/datafiles/esc_eval_data_'+ str(fold) +'.json', 'w') as f:
    #     json.dump({'data': eval_wav_list}, f, indent=1)

print('Finished UrbanSound8K Preparation')