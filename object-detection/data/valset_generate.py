'''
To generated valset and new train set from  original train set.
'''
import os
import shutil 
import json

def valset_gen(root, destination, train_json_path, valset_size):
    
    src_path = os.path.join(root, 'train')
    destination = os.path.join(root, destination)
    img_list = os.listdir(src_path)
    val_list = img_list[:valset_size]
    
    # Validation set generation
    if not os.path.exists(destination):
        os.makedirs(destination)
    
    for img in val_list:
        shutil.move(os.path.join(src_path, img), destination)
    
    valset_list = os.listdir(destination)
    if len(valset_list) == 1000:
        print('Validation images are successfully generated.')
    
    json_data = json.load(open(os.path.join(root, train_json_path)))
    
    val_dict = {}
    for im_name in valset_list:
        val_dict[im_name] = json_data[im_name]
    
    with open(os.path.join(root, 'val_annotations.json'), 'w') as outfile:
        json.dump(val_dict, outfile) 
    print('Validation labels are successfully generated.')
    
    # Training set after moving validation images
    new_train_list = os.listdir(src_path)
    
    new_train_dict = {}
    for im_name in new_train_list:
        new_train_dict[im_name] = json_data[im_name]
        
    with open(os.path.join(root,'new_train_annotations.json'), 'w') as outfile:
        json.dump(new_train_dict, outfile) 
    print('New training labels are successfully generated.')

if __name__ == "__main__":
    
    valset_gen(root='DelftBikes', destination='val', train_json_path='train_annotations.json', valset_size=1000)