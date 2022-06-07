# VIPriors Action Recognition Challenge

*Maintainer: Marcos Baptista Ríos (mbaptista@alicebiometrics.com)*

Welcome to the VIPriors Action Recognition Challenge. This challenge is part of the "[3rd Visual Inductive Priors for Data-Efficient Deep Learning Workshop](https://vipriors.github.io/)", which will be held at [ECCV 2022](https://eccv2022.ecva.net).

The common idea of all VIPriors challenges is to have models trained from scratch on an reduced version of a certain dataset for a certain Computer Vision task. In this particular challenge, the task is Action Recognition and the original dataset is Kinetics400.

## Data

We provide a modification of the Kinetics400 dataset with train, validation and test subsets. Please:

- **Do not use the original distribution of the Kinetics400 dataset.** 
- **DO NOT PRETRAIN your models ON ANY DATASET.** 
- **TRAIN your model FROM SCRATCH.** 

Please, find more information on how to get the modified dataset as well as the annotations in the data README.

## Evaluation

The task will be evaluated using the mean average accuracy over all classes on the test set. The winner of the challenge will be determined with the highest Top-1 accuracy. However, as extra information of the models, we will also compute the Top-3 and Top-5 accuracy.

[Please find the evaluation server here.](https://codalab.lisn.upsaclay.fr/competitions/4703)

For more information about the evaluation, please refer to the evaluation toolkit README.

## Additional information

All the tooling provided for this challenge was tested using a common python 3.7 installation. The only needed packages are numpy, pandas and and scikit-learn. You can find the models in the `requirements.txt `file. Although it should work with different versions, we recommend to use the same which we tested with.

## Questions

If you find any problem/bug or have any doubt about the VIPriors Action Recognition Challenge, please contact with the maintainer of this challenge at mbaptista@alicebiometrics.com
