# VIPriors Action Recognition Challenge - Data

The VIPRiors Action Recognition Challenge uses a modification of the [Kinetics400](https://deepmind.com/research/open-source/kinetics) action recognition dataset. Please:

- **Do not use the original distribution of the Kinetics400 dataset.**
- **DO NOT PRETRAIN your models on any dataset.**
- **TRAIN your model FROM SCRATCH.**

Violating this restrictions may result in disqualification. Additionally, the challenge loses its purpose if participants do not meet the rules.

## Setting up the data

We know the difficulties of downloading and arranging such a large dataset as Kinetics400. For this reason, we provide a link to directly download the modified version that we use in this challenge. **Please, use this link to download it [through SURFDrive (~100GB)](https://surfdrive.surf.nl/files/index.php/s/fQ41gfR0ifgjW2A)**.

Additionally, we also provide links in the file `Kinetics400ViPriors_urls.txt` to download it through Google Storage. You just have to do as follows:

`wget -i Kinetics400ViPriors_urls.txt`

`cat *.tar.gz.* | tar xvfz -`

Once the challenge dataset (**Kinetics400ViPriors**) has been downloaded, you will have a directory with the following structure:

```
Kinetics400ViPriors
├── annotations
│   ├── kinetics400ViPriors-train.csv
│   ├── kinetics400ViPriors-val.csv
│   ├── kinetics400ViPriors-test-public.csv
│   └── clsIdx.csv
├── train
├── val
└── test
```

Description of the content:

- `annotations/clsInd.csv`: class label and index.
- `annotations/kinetics400ViPriors-train.csv`: annotated ground truth of the training videos.
- `annotations/kinetics400ViPriors-val.csv`: annotated ground truth of the validation videos.
- `annotations/kinetics400ViPriors-test-public.csv`: test videos, without annotations.
- `Kinetics400ViPriors_urls.txt`: urls to download dataset.
- `train/*.mp4`: training video files.
- `val/*.mp4`: validation video files.
- `test/*.mp4`: test video files.

Finally, you wil have:

- Train set: 40K clips.
- Validation set: 10K clips.
- Test set: 20K clips.
