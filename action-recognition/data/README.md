# VIPriors Action Recognition Challenge - Data

The VIPRiors Action Recognition Challenge uses a modification of the [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) action recognition dataset. Please:

- **Do not use the original distribution of the UCF101 dataset.** 
- **DO NOT PRETRAIN your models on any dataset.** 
- **TRAIN your model FROM SCRATCH.** 

Violating this restrictions may result in disqualification. Additionally, the challenge loses its purpose if participants do not meet the rules.

## Setting up the data

The tooling provided automatically rearrange the original dataset to the modified version with its ground truth files (only labels for train and validation are provided). Follow the instructions below to set up the data for this challenge:

1. Download the original dataset from [here](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar).
2. Extract the content of the .rar file to this directory. You have to see the UCF101 directory with all action directories inside. If, for whatever reason, the extracted folder has a different name, please rename it to UCF101. This is the directory the data toolkit will look for.
3. Download the original annotations from [here](https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip). and put the .zip file in this directory.
4. Run "arrange_data.py" script: `python arrange_data.py` (it may take a while to finish)

Once the script has been executed, you will have a new directory `mod-ucf101/` with the following structure:

```
mod-ucf101
├── annotations
│   ├── mod-ucf101-classInd.txt
│   ├── mod-ucf101-test.txt
│   ├── mod-ucf101-train.txt
│   └── mod-ucf101-validation.txt
└── videos
```

Description of the content:

- `annotations/mod-ucf101-classInd.txt`: class label and index.
- `annotations/mod-ucf101-train.txt`: annotated ground truth of the training videos.
- `annotations/mod-ucf101-validation.txt`: annotated ground truth of the validation videos.
- `annotations/mod-ucf101-test.txt`: test videos, without annotations.
- `videos/video_{train/validation/test}_{0000X}.avi`: video files.

Finally, you wil have:

- Train set: ~4.8K clips.
- Validation set: ~4.7K clips.
- Test set: ~3.8K clips.