"""
Arrange training, validation and test ground truth files for VIPriors Action
Recognition Challenge.
"""
import os
from zipfile import ZipFile
import math
import shutil
import random


if __name__ == '__main__':
    zip_file = './UCF101TrainTestSplits-RecognitionTask.zip'
    ucf_dir = './UCF-101/'
    new_ann_dir = './mod-ucf101/annotations/'
    new_videos_dir = './mod-ucf101/videos/'
    new_cls_file = f'{new_ann_dir}mod-ucf101-classInd.txt'
    new_train_file = f'{new_ann_dir}mod-ucf101-train.txt'
    new_val_file = f'{new_ann_dir}mod-ucf101-validation.txt'
    new_test_file = f'{new_ann_dir}mod-ucf101-test.txt'

    if not os.path.isfile(zip_file):
        raise FileExistsError('Original annotation .zip file was not found. '
                              'Please, make sure to put the original file in '
                              'the current directory and with its original '
                              'name.')

    with ZipFile(zip_file, 'r') as zipObj:
        zipObj.extractall()

    print('Generating VIPriors mod-UCF101...')

    if not os.path.exists(new_ann_dir):
        os.makedirs(new_ann_dir)

    cls_lst = list()
    ind_lst = list()
    with open('./ucfTrainTestlist/classInd.txt', 'r') as cls_f:
        cls_ind = cls_f.readlines()

    for line in cls_ind:
        ind_lst.append(line.split(' ')[0])
        cls_lst.append(line.split(' ')[1].split('\n')[0])

    new_ind_lst = ind_lst[:]
    random.seed(1992)
    random.shuffle(new_ind_lst)

    with open(new_cls_file, 'w') as f:
        for i, val in enumerate(new_ind_lst):
            f.write(f'{new_ind_lst[i]} {cls_lst[i]}\n')

    with open('./ucfTrainTestlist/trainlist01.txt', 'r') as train_f:
        lines = train_f.readlines()

    orig_gt_train = dict()
    for c in cls_ind:
        cls = c.split(' ')[-1].split('\n')[0]
        orig_gt_train[cls] = list()
        for l in lines:
            this_l_cls = l.split('/')[0]
            if this_l_cls == cls:
                orig_gt_train[cls].append(l.split('\n')[0])
            else:
                continue

    mod_ucf101_gt_train = dict()
    mod_ucf101_gt_val = dict()
    if not os.path.exists(new_videos_dir):
        os.mkdir(new_videos_dir)
    for c, v in orig_gt_train.items():
        n_train_vid = math.ceil(len(v) / 2)
        mod_ucf101_gt_train[c] = v[0:n_train_vid]
        mod_ucf101_gt_val[c] = v[n_train_vid:]

    vid_idx = 0
    with open(new_train_file, 'w') as new_train_f:
        for c, v in mod_ucf101_gt_train.items():
            for i, clip in enumerate(v):
                vid_idx = vid_idx + 1
                name = clip.split(' ')[0]
                idx = int(clip.split(' ')[1])
                new_name = f'video_train_{vid_idx:07}.avi'
                shutil.copy(f'{ucf_dir}{name}', f'{new_videos_dir}{new_name}')
                new_train_f.write(f'{new_name} {new_ind_lst[idx - 1]}\n')

    vid_idx = 0
    with open(new_val_file, 'w') as new_val_f:
        for c, v in mod_ucf101_gt_val.items():
            for i, clip in enumerate(v):
                vid_idx = vid_idx + 1
                name = clip.split(' ')[0]
                idx = int(clip.split(' ')[1])
                new_name = f'video_validation_{vid_idx:07}.avi'
                shutil.copy(f'{ucf_dir}{name}', f'{new_videos_dir}{new_name}')
                new_val_f.write(f'{new_name} {new_ind_lst[idx - 1]}\n')

    with open('./ucfTrainTestlist/testlist01.txt', 'r') as test_f:
        lines = test_f.readlines()

    orig_gt_test = dict()
    for c in cls_ind:
        cls = c.split(' ')[-1].split('\n')[0]
        orig_gt_test[cls] = list()
        for l in lines:
            this_l_cls = l.split('/')[0]
            if this_l_cls == cls:
                orig_gt_test[cls].append(l.split('\n')[0])
            else:
                continue

    vid_idx = 0
    with open(new_test_file, 'w') as new_test_f:
        for c, v in orig_gt_test.items():
            for i, clip in enumerate(v):
                vid_idx = vid_idx + 1
                name = clip.split(' ')[0]
                new_name = f'video_test_{vid_idx:07}.avi'
                shutil.copy(f'{ucf_dir}{name}', f'{new_videos_dir}{new_name}')
                new_test_f.write(f'{new_name}\n')

    shutil.rmtree('./ucfTrainTestlist')
    shutil.rmtree(ucf_dir)

    print('Dataset created. Enjoy the challenge!')
