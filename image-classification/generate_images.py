"""
Arrange training, validation and test data for VIPriors image classification
challenge.
"""

import os
import shutil
import random
import sys
import time
from tqdm import tqdm
import glob
import pickle

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

if __name__ == '__main__':
    main_dir_path = sys.argv[1] #directory of train set "/home/user/imagenet/raw-data"
    output_dir = sys.argv[2] #output location "/home/user/datasets/"
    main_traindir_path = main_dir_path +'/'+ 'train'
    main_valdir_path = main_dir_path +'/'+ 'val'
    
    train_dir = output_dir + 'imagenet_50/train/'
    if not os.path.exists(train_dir):
           print("path doesn't exist. trying to make")
           os.makedirs(train_dir)
    
    val_dir = output_dir + 'imagenet_50/val/'
    if not os.path.exists(val_dir):
           print("path doesn't exist. trying to make")
           os.makedirs(val_dir)
           
    test_dir = output_dir + 'imagenet_50/test/imgs'
    if not os.path.exists(test_dir):
           print("path doesn't exist. trying to make")
           os.makedirs(test_dir)
    t1 = time.time()
    
    dirnames = listdir_fullpath(main_traindir_path)
    test_dirnames = listdir_fullpath(main_valdir_path)
    
    pickle_in = open("train.pickle","rb")
    train = pickle.load(pickle_in)
    pickle_in.close()
    pickle_in = open("val.pickle","rb")
    val = pickle.load(pickle_in)
    pickle_in.close()
    pickle_in = open("test.pickle","rb")
    test = pickle.load(pickle_in)
    pickle_in.close()
    
    
    
    for image_dir in tqdm(dirnames):
        train_dir_name = train_dir + os.path.basename(image_dir)
        val_dir_name = val_dir + os.path.basename(image_dir)
    
        if os.path.exists(train_dir_name):    
            shutil.rmtree(train_dir_name)
            pass
        os.mkdir(train_dir_name)
        if os.path.exists(val_dir_name):    
            shutil.rmtree(val_dir_name)
            pass
        os.mkdir(val_dir_name)
    
        for train_file, val_file in zip(train,val):
                if image_dir == main_traindir_path + '/'+train_file[0]:
                    train_file_name = train_dir_name + "/" + os.path.basename(train_file[1])
                    shutil.copy2(main_traindir_path+ '/'+train_file[0] + '/'+train_file[1], train_file_name)
                    
                    val_file_name = val_dir_name + "/" + os.path.basename(val_file[1])
                    shutil.copy2(main_traindir_path+ '/'+train_file[0] + '/'+val_file[1], val_file_name)
                else:
                    continue
                
    for idx, old_name in tqdm( enumerate(test)):
        test_file_name = test_dir + "/" + os.path.basename(main_valdir_path+'/'+old_name)
        shutil.copy2(main_valdir_path+'/'+old_name, test_file_name )
        new_name = test_dir + "/" + "img_" + str(idx) + '.JPEG'
        os.rename(test_file_name, new_name)
    
    t2 = time.time()
    
    print("Data generation time : ", (t2-t1)/ 60.0)
