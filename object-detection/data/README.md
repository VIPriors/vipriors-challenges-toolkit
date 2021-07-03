# VIPriors Object Detection challenge - Data

The VIPriors object detection challenge uses [DelftBikes](https://github.com/oskyhn/DelftBikes) dataset. In particular, we use the bounding box annotations.

Please follow the instructions below to set up the data for this challenge. To submit your contribution to our challenge generate your models predictions over the test set and submit the predictions according to the instructions in the [main README](../README.md).

We also provide a validation set derived from training set. Validation results can be submitted to **Development(Validation set)**.
For final submission, you can use both **training** and **validation** sets for training.

## Setting up data

These are the instructions for setting up the data for the VIPriors object detection challenge.

1. Download the DelftBikes images from [DelftBikes](https://github.com/oskyhn/DelftBikes);
2. Extract the ZIP file to `data`. This creates folders `data/train`, `data/test`, `train_annotations.json` and `fake_test_annotations.json`. `fake_test_annotations.json` is used just for generating submission.

Now you are ready to use the data.

- The root directory of the training and testing images is `data/DelftBikes`;
- The annotations files to be used are:
  - Train set (8K images): `data/annotations/train_annotations.json`
  - Fake Testing set (2K images, no correct labels provided): `data/annotations/fake_test_annotations.json`
  - To generate validation and new train set (both are derived from original training set), you can run `valset_generation.py` which is located under `data` directory. The script moves 1000 train images to `val` directory and produces new annotations:
    - `val` directory consists of 1000 images.
    - `train` directory consists of 7000 images.
    - `val_annotations.json` and `new_train_annotations.json`.
