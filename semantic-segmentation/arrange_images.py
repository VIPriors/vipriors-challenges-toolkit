#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 18:39:52 2020 by Attila Lengyel - a.lengyel@tudelft.nl
"""

import tqdm
import os
import pickle
import argparse
import shutil

parser = argparse.ArgumentParser(description='PyTorch ECP Object Detection Training')

parser.add_argument('cityscapes_dir', metavar='path/to/cityscapes/root',
                    help='root dir of cityscapes dataset')

def main():
    args = parser.parse_args()
    
    # Define output dir
    output_dir = 'minicity'
    
    # Check if output dir exists
    if not os.path.exists(output_dir):
        for subset in ['train', 'val']:
            imdir = os.path.join(output_dir,'leftImg8bit', subset)
            gtdir = os.path.join(output_dir,'gtFine', subset)
            # Create output dirs
            os.makedirs(imdir)
            os.makedirs(gtdir)
            
            with open(os.path.join('helpers',subset+'.p'), 'rb') as f:
                files = pickle.load(f)
            
            for f in tqdm.tqdm(files, desc='Generating '+subset+' subset'):
                filename = os.path.split(f)[-1]
                # Copy image
                shutil.copy(os.path.join(args.cityscapes_dir,'leftImg8bit/train',f),
                            os.path.join(imdir,filename))
                # Copy labels
                gtinst = f[:-15]+'gtFine_instanceIds.png'
                gtlbl = f[:-15]+'gtFine_labelIds.png'
                shutil.copy(os.path.join(args.cityscapes_dir,'gtFine/train',gtinst),
                            os.path.join(gtdir,os.path.split(gtinst)[-1]))
                shutil.copy(os.path.join(args.cityscapes_dir,'gtFine/train',gtlbl),
                            os.path.join(gtdir,os.path.split(gtlbl)[-1]))
        # Create output dirs
        os.makedirs(os.path.join(output_dir,'leftImg8bit/test'))
        
        with open(os.path.join('helpers','test.p'), 'rb') as f:
            files = pickle.load(f)
        
        for f in tqdm.tqdm(files, desc='Generating test subset'):
            # Copy image
            shutil.copy(os.path.join(args.cityscapes_dir,'leftImg8bit/val',f),
                        os.path.join(output_dir,'leftImg8bit/test',str(files[f])+'.png'))
    else:
        raise AssertionError('Dataset already generated (output dir exists).')
        
if __name__ == "__main__":
    main()