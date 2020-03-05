# VIPriors Object Detection challenge - Data

The VIPriors object detection challenge uses a subset of the 2017 MS COCO object detection dataset. In particular, we use the bounding box annotations.

Please note that we repurpose the original 2017 validation set as a test set. **Do not use the original MS COCO validation set in training or validating your model!** Any usage of the validation set in constructing your model is in violation of the challenge rules and may result in disqualification.

Please follow the instructions below to set up the data for this challenge. The tooling provided automatically arranges the data into appropriate training, validation and testing sets. To submit your contribution to our challenge generate your models predictions over the test set and submit the predictions according to the instructions in the [main README](../README.md).

## Setting up data

These are the instructions for setting up the data for the VIPriors object detection challenge.

1. Download the 2017 train/val images from MS COCO website;
2. Extract both ZIPs to `data`. This creates folders "train2017" and "val2017".
3. Download the VIPriors object detection annotations from [the CodaLab competition website](https://competitions.codalab.org/competitions/23661#participate-get_starting_kit) and extract the ZIP file in `data` to create folder `data/annotations` and file `test_image_mappings.txt`.
4. Run `python arrange_images.py` from folder `data` to automatically prepare the images according to challenge instructions.

Now you are ready to use the data.

- The root directory of the training and validation images is `data/images`;
- The root directory of the testing images is `data/test-images`;
- The annotations files to be used are:
  - Train set (~23K images): `data/annotations/vipriors-object-detection-train.json`
  - Validation set (5K images): `data/annotations/vipriors-object-detection-val.json`
  - Testing set (5K images, no labels provided): `data/annotations/vipriors-object-detection-test.json`
