# VIPriors Action Recognition Challenge - Data

The VIPRiors Action Recognition Challenge uses **Kinetics400ViPriors**, a modification of the [Kinetics400](https://deepmind.com/research/open-source/kinetics) action recognition dataset. Please:

- **DO NOT USE THE ORIGINAL distribution of the Kinetics400 dataset.**
- **DO NOT PRETRAIN your models ON ANY DATASET.**
- **TRAIN your model FROM SCRATCH.**

Violating this restrictions may result in disqualification. Additionally, the challenge loses its purpose if participants do not meet the rules.

## Setting up the data
Kinetics400ViPriors contains ~100GB of videos. We know the difficulties of downloading and arranging large video datasets. For this reason, we provide two ways to get our dataset: **SURFDrive** and **Google Storage**.

### Download through SURFDrive
You can directly download the dataset from the files you have in this link: [SURFDrive](https://surfdrive.surf.nl/files/index.php/s/fQ41gfR0ifgjW2A).

### Download through Google Storage
We have created a Google Storage bucket from which you can download Kinetics400ViPriors in several parts that will have to be joined afterwards. The file `Kinetics400ViPriors_urls.txt` that you can find in this directory contains the links to the different parts. You just have to do as follows:

```bash
wget -i Kinetics400ViPriors_urls.txt
cat *.tar.gz.* | tar xvfz -
```

### Check your downloaded files
Once you have downloaded Kinetics400ViPriors please make yourself sure whether your data is at the same version of the challenge. To do this, we provide code. Just run the following:
```bash
python kinetics400ViPriors_data_check.py
```
This will check if you have all available videos. And it also checks if you do not have any other corrupted video, apart from those we currently consider. The list of the current blocked videos is in `annotations/kinetics400ViPriors_corrupted_videos.txt`

### Content of this directory
Once **Kinetics400ViPriors** has been downloaded, this directory should have the following structure:

```
Kinetics400ViPriors
├── annotations
│   ├── kinetics400ViPriors-train.csv
│   ├── kinetics400ViPriors-val.csv
│   ├── kinetics400ViPriors-test-public.csv
│   ├── kinetics400ViPriors_corrupted_videos.txt
│   └── clsIdx.csv
├── train
├── val
└── test
```
With more detail, you will have:
- `annotations/clsInd.csv`: class label and index.
- `annotations/kinetics400ViPriors-train.csv`: annotated ground truth of the training videos.
- `annotations/kinetics400ViPriors-val.csv`: annotated ground truth of the validation videos.
- `annotations/kinetics400ViPriors-test-public.csv`: test videos, without annotations.
- `annotations/Kinetics400ViPriors_corrupted_videos.txt`: list of current blocked videos (from all subsets).
- `Kinetics400ViPriors_urls.txt`: urls to download dataset.
- `train/*.mp4`: training video files.
- `val/*.mp4`: validation video files.
- `test/*.mp4`: test video files.

Make sure you have: 
- Train set: ~40K clips.
- Validation set: ~10K clips.
- Test set: ~20K clips.
