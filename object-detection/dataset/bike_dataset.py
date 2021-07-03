#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
import json
from PIL import Image

class DelftBikeDataset(object):
    def __init__(self, root, json_path, transforms, mode='train'):
        self.root = root
        self.transforms = transforms
        self.json_path = json_path
        self.mode = mode
        self.image_path = self.mode 

        self.imgs = list(sorted(os.listdir(os.path.join(self.root, self.image_path))))
        self.json_data = json.load(open(os.path.join(self.root, self.json_path)))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, self.image_path, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        labels = self.json_data[self.imgs[idx]]
        # get bounding box coordinates for each mask
        num_objs = len(labels['parts'])
        boxes = []
        labs = []
    
        for ind,i in enumerate(labels['parts'],0):
            lab = labels['parts'][i]
            if  lab['object_state'] != 'absent':
                loc = lab['absolute_bounding_box']
                xmin = loc['left']
                xmax = loc['left'] + loc['width']
                ymin = loc['top']
                ymax = loc['top'] + loc['height']
                boxes.append([xmin, ymin, xmax, ymax])
                labs.append(ind+1) 
        # convert everything into a torch tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labs = torch.as_tensor(labs, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labs
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)