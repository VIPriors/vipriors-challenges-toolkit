# VIPriors Semantic Segmentation challenge

*Maintainer: Attila Lengyel (a.lengyel@tudelft.nl)*

We present the "Visual Inductive Priors for Data-Efficient Computer Vision" challenge. We offer four challenges, where models are to be trained from scratch, and we reduce the number of training samples to a fraction of the full set. The winners of each challenge are invited to present their winning method at the VIPriors workshop presentation at ECCV 2020.

This challenge is the semantic segmentation challenge. We provide a subset of the [Cityscapes dataset](https://www.cityscapes-dataset.com) (MiniCity) to train on. We will evaluate all models submitted to the challenge on Cityscapes validation data.

## Datasets

The task to be performed is semantic segmentation. The training and validation data are subsets of the training split of the Cityscapes dataset. The test set is taken from the validation split of the Cityscapes dataset.

As a note: **Do not use the original Cityscapes validation set in training or validating your model!** Any usage of the validation set in constructing your model is in violation of the challenge rules and may result in disqualification.

## Setup

#### Dataset

Please follow the instructions below to set up the data for this challenge. The tooling provided automatically arranges the data into appropriate training, validation and testing sets.

These are the instructions for setting up the data for the VIPriors object detection challenge.

1. Download the `gtFine_trainvaltest.zip` (241MB) and `leftImg8bit_trainvaltest.zip` (11GB) files from the [Cityscapes website](https://www.cityscapes-dataset.com/downloads/) (login required);
2. Extract the ZIP files to a directory containing the `gtFine` and `leftImg8bit` folders.
3. Generate the dataset subset by running `python arrange_images.py 'path/to/cityscapes/directory'`. This creates the `minicity` directory containing the small dataset.

Now you are ready to use the data. You can easily load the dataset into PyTorch using the dataset definition in `helpers/minicity.py`. See `baseline.py` for example usage.

#### Training and prediction

You can now run `python baseline.py` to train the baseline model as described below and perform prediction on the validation set. This creates a `baseline_run` directory containing all logs and other files generated during training and a `results` directory with the final predictions on the validation set.

#### Evaluation

The evaluation criteria are the same as used for the [Cityscapes Pixel-Level Semantic Labeling Task](https://www.cityscapes-dataset.com/benchmarks/#scene-labeling-task). The main metric used to rank submissions is the mean Intersection-over-Union (mIoU). Please refer to the [class definitions as described here](https://www.cityscapes-dataset.com/dataset-overview/#class-definitions) to see which classes are included in the evaluation.

We provide an evaluation script to test your model over the validation set. Note that this script cannot be used to evaluate models over the testing set, as we do not provide labels for the test set. It is good practice to ensure your predictions work with this script, as the same script is used on the evaluation server.

Run `python evaluate.py` to evaulate the predictions in your `results` directory. This creates a `results.txt` files containing the metrics used for the challenge leaderboard and a `results.json` file containing a more detailed evaluation.

## Submissions

The evaluation server is hosted using CodaLab. Submitting to the challenge requires a CodaLab account.

~~Please find the evaluation server here~~. *The evaluation server will soon be opened.*

To participate in the challenge one uploads a .zip file to the evaluation server containing all predictions over the challenge test set in .png format. This is essentially a zipped version of the files in the `results` directory. Please ensure that your .zip file only contains the .png.

## Baselines

We provide a PyTorch implementation of [U-Net](https://arxiv.org/abs/1505.04597) with Batch Normalization. To train and evaluate the model on the MiniCity dataset, run `python baseline.py`. Note: the scripts assumes that a GPU is available in your system. Training for 200 epochs takes approximately 2:15 hours on a system with a GeForce GTX 1080 Ti GPU.

The baseline performance on the provided validation set is given in the table below:

| U-Net w/ Batch Normalization | checkpoint_path |
| ---------------------------- | --------------- |
| IoU Class                    | 0.36            |
| iIoU Class                   | 0.16            |
| IoU Category                 | 0.71            |
| iIou Category                | 0.53            |
| Accuracy                     | 0.77            |
