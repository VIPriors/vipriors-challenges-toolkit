# Object detection reference training scripts

*NOTE: this code is originally from the [Torchvision object detection finetuning tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html). It has been modified in places to accomodate our needs. Please see the license terms in the top directory of the repository.*

This folder contains reference training scripts for object detection.
They serve as a log of how to train specific models, as provide baseline
training and evaluation scripts to quickly bootstrap research.

Except otherwise noted, all models have been trained on 8x V100 GPUs.

### Faster R-CNN
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model fasterrcnn_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3
```


### Mask R-CNN
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco --model maskrcnn_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3
```


### Keypoint R-CNN
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --dataset coco_kp --model keypointrcnn_resnet50_fpn --epochs 46\
    --lr-steps 36 43 --aspect-ratio-group-factor 3
```