# Setting up data

These are the instructions for setting up the data for the VIPriors object detection challenge.

1. Download the 2017 train/val images from MS COCO website;
2. Extract both ZIPs to `/data`. This creates folders "train2017" and "val2017".
3. Merge the "train2017" and "val2017" folders in one folder "images", which now resides at `/data/images`.
4. Download the VIPriors object detection annotations from [this link](https://vipriors.githiub.io/challenges and extract the ZIP file in `/data` to create folder `/data/annotations`.

Now you are ready to use the data.

- The root directory of the images should be `/data/images`;
- The annotations files to be used are:
  - Train set (~23K images): `/data/annotations/vipriors-object-detection-train.json`
  - Validation set (5K images): `/data/annotations/vipriors-object-detection-val.json`
  - Testing set (5K images, no labels provided): `/data/annotations/vipriors-object-detection-test.json`
