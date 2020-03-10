# VIPriors Object Detection challenge

*Maintainer: Robert-Jan Bruintjes (r.bruintjes@tudelft.nl)*

We present the "Visual Inductive Priors for Data-Efficient Computer Vision" challenge. We offer four challenges, where models are to be trained from scratch, and we reduce the number of training samples to a fraction of the full set. The winners of each challenge are invited to present their winning method at the VIPriors workshop presentation at ECCV 2020.

This challenge is the object detection challenge. We provide a subset of the MS COCO dataset to train on. We will evaluate all models submitted to the challenge on MS COCO validation data.

## Datasets

The task to be performed is object detection, predicting bounding boxes. The training and validation data are subsets of the training split of the MS COCO dataset (2017 release, bounding boxes only). The test set is taken from the validation split of the MS COCO dataset.

As a note: **DO NOT train on MS COCO validation data.** Please use the tooling described here to set up your training, validation and test data to avoid accidentally training on test data.

To find instructions on setting up data please refer to [the data README](data/README.md).

## Validation

We provide an evaluation script to test your model over the validation set. Note that this script cannot be used to evaluate models over the testing set, as we do not provide labels for the test set. It is good practice to ensure your predictions work with this script, as the same script is used on the evaluation server.

## Submissions

The evaluation server is hosted using CodaLab. Submitting to the challenge requires a CodaLab account.

[Please find the evaluation server here.](https://competitions.codalab.org/competitions/23661)

To participate in the challenge one uploads a file of predictions over the challenge test set to the evaluation server. Generate these predictions by inferring your model over our test set (see [data README](data/README.md) for how to get these images) and using the provided `submission_format.py` module to store the predictions. You can also refer to the baseline code in `baseline.py`, which includes example code on how to store prediction results as a submission file.

The submissions file is a JSON encoding of a list of bounding box predictions. This is the format of a submission file:

```json
[
    {
        "image_id": "0.jpg",
        "category_id": 5,
        "bbox": [220,220,30,30],
        "score": 0.98,
    },
    ...
]
```

## Baselines

We trained a simple baseline model: a Faster R-CNN model with ResNet-18 FPN backbone, trained from scratch for 51 epochs with initial learning rate `0.02` and decay at epoch 48.

| **model**           | **checkpoint** | **Test set AP @ 0.50:0.95** |
| ------------------- | -------------- | ---------------- |
| Faster R-CNN ResNet-18 FPN | [Download checkpoint](https://competitions.codalab.org/my/datasets/download/bc13517e-5ef7-4dda-b649-2d6a0d62a7eb)      | 0.049       |

The baseline models may be used as checkpoints from which to fine-tune. Please note this is the **only exception** to the rule which forbids fine-tuning.

### Training or fine-tuning baselines

*NOTE: this code is originally from the [Torchvision object detection finetuning tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html). It has been modified in places to accomodate our needs. Please see the license terms in the top directory of the repository.*

These are example commands to use to train or fine-tune the provided baselines.

**Faster R-CNN - ResNet-18 FPN backbone**

Execute this from the `object-detection` folder:

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train_baseline.py\
    --dataset coco --model fasterrcnn_resnet18_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3
```