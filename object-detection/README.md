# VIPriors Object Detection challenge

*Maintainer: Osman Semih Kayhan (o.s.kayhan@tudelft.nl)*

We present the "3rd Visual Inductive Priors for Data-Efficient Computer Vision" challenge. We offer four challenges, where models are to be trained from scratch. The winners of each challenge are invited to present their winning method at the VIPriors workshop presentation at ECCV 2022.

This challenge is the object detection challenge. We provide [DelftBikes](https://github.com/oskyhn/DelftBikes) dataset to train on. We will evaluate all models submitted to the challenge test data.

## Datasets

The task to be performed is object detection, predicting bounding boxes. [DelftBikes](https://github.com/oskyhn/DelftBikes) contains 10,000 bike images with 22 densely annotated parts for each bike. Besides, we explicitly annotate all part locations and part states as missing, intact, damaged, or occluded. To note that, the dataset contains some noisy labels too, thefore it is more challenging. The evaluation is done on avaliable part, namely intact, damaged and occluded parts. For more information about dataset, you can check the [paper](https://arxiv.org/abs/2106.02523).

We also provide a validation set which derived from training set. Validation results can be submitted to **Development (Validation set)**.

For final submission, you can use both **training** and **validation** sets for training. We provide train labels and fake test labels to be able to generate submission. To note that, evaluation is done on images with their **original sizes**.

To find instructions on setting up data please refer to [the data README](data/README.md).


## Submissions

The evaluation server is hosted using CodaLab. Submitting to the challenge requires a CodaLab account.

[Please find the evaluation server here.](https://codalab.lisn.upsaclay.fr/competitions/4696)

To participate in the challenge one uploads a file of predictions over the challenge test set to the evaluation server. Generate these predictions by inferring your model over our test set (see [data README](data/README.md) for how to get these images) and using the provided script to store the predictions. You can also refer to the baseline code in `train_baseline.py`, which includes example code on how to train Faster RCNN network. To generate and store prediction results as a submission file please use `generate_submission.py`.

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

We trained a simple baseline model: a Faster R-CNN model with ResNet-50 FPN backbone, trained from scratch for 16 epochs with initial learning rate `0.01` and decay at epoch 15.
Image sizes are kept as original sizes and evaluation is also done on **original sizes**. Baseline model does not have any data augmentation.  The default hyperparameters are tuned for training on a single GPU and with batch size of 4.

| **model**           | **checkpoint** | **Test set AP @ 0.50:0.95** |
| ------------------- | -------------- | ---------------- |
| Faster R-CNN ResNet-50 FPN | [Download checkpoint](https://competitions.codalab.org/my/datasets/download/2c4c6b4a-c50d-4c38-89d4-0da551d20a81)      | 0.258       |

The baseline models may be used as checkpoints from which to fine-tune. Please note this is the **only exception** to the rule which forbids fine-tuning.

### Training or fine-tuning baselines

*NOTE: this code is originally from the [Torchvision object detection finetuning tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html). It has been modified in places to accomodate our needs. Please see the license terms in the top directory of the repository.*

These are example commands to use to train or fine-tune the provided baselines.

**Faster R-CNN - ResNet-50 FPN backbone**

Execute this from the `object-detection` folder:

```
python train_baseline.py --data_path </data/DelftBikes/> \
 --train_json <train_annotations.json> 
```
To generate submission:

- Test submission:
```
python generate_submission.py  --data_path </data/DelftBikes/>\
 --test_json <fake_test_annotations.json> --resume <checkpoint>\

```
- Val submission:
```
python generate_submission.py  --data_path </data/DelftBikes/>\
 --eval_mode val --test_json <val_annotations.json> --resume <checkpoint>\

```
