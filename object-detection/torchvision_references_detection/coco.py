from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
import torch
import numpy as np


# TODO: remove `device`?
def convert_ann(ann, device):
    xmin, ymin, w, h = ann['bbox']
    # DEBUG
    if w <= 0 or h <= 0:
        raise ValueError("Degenerate bbox (x, y, w, h): ", str([xmin, ymin, w, h]))
    ann['boxes'] = [xmin, ymin, xmin + w, ymin + h]
    ann['labels'] = ann['category_id']

    del ann['bbox']
    del ann['category_id']
    del ann['segmentation']

    return ann

def flatten_targets(targets):
    """
    Pre-process annotations to match format expected by torchvision's
    GeneralizedRCNN.
    """
    if len(targets) == 0:
        return {}

    # Flatten annotations into one dictionary
    flat_obj = {}
    for k in targets[0].keys():
        if k == 'image_id':
            # Image ID is special: we only want one of them, regardless of
            # the amount of annotations
            flat_obj[k] = np.array(targets[0][k])
            # print("image ID: ", flat_obj[k])
        else:
            # print(k)
            flat_obj[k] = np.stack([t[k] for t in targets])
    return flat_obj

class CocoDetection(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, annFile, allow_missing_targets=True, transform=None, target_transform=None, transforms=None):
        super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.allow_missing_targets = allow_missing_targets

        # DEBUG: convert format of all COCO annotations to training/testing format
        print("converting to PyTorch format...")
        for id in self.coco.anns:
            self.coco.anns[id] = convert_ann(self.coco.anns[id], 'cpu')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        if len(ann_ids) == 0:
            if not self.allow_missing_targets:
                raise ValueError("Image has no annotations!")
            # DEBUG
            target = [{'image_id': img_id}]

        # DEBUG
        target = flatten_targets(target)

        path = coco.loadImgs(img_id)[0]['file_name']

        with Image.open(os.path.join(self.root, path)) as img:
            img.convert('RGB')

            if self.transforms is not None:
                img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)