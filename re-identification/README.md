# Re-Identification task for the VIPriors Challenge

Manteiner: Davide Zambrano from Synergy Sports (davide.zambrano@synergysports.com)

**NOTE this code is based on Open-reid repo: https://github.com/Cysu/open-reid.git"**

_Open-ReID is a lightweight library of person re-identification for research
purpose. It aims to provide a uniform interface for different datasets, a full
set of models and evaluation metrics, as well as examples to reproduce (near)
state-of-the-art results._

We want to thank the authors for providing this tool. This version applies some changes to the original code to specifically adapt it to the VIPrior Challenge on Person Re-Identification. 

We present the "Visual Inductive Priors for Data-Efficient Computer Vision" challenge. This year, we offer five challenges, where models are to be trained from scratch, and we reduce the number of training samples to a fraction of the full set. The winners of each challenge are invited to present their winning method at the VIPriors workshop presentation at ICCV 2021.

## Installation

**Note that the file ```setup.py``` specifies the libraries version to use to run the code.**

Install [PyTorch](http://pytorch.org/). 

```shell
git clone https://github.com/VIPriors/vipriors-challenges-toolkit
cd vipriors-challenges-toolkit/re-identification
pip install -e .
```

## Examples

```shell
python baseline/synergyreid_baseline.py -b 64 -j 2 -a resnet50 --logs-dir logs/synergy-reid/
```

This is just a quick example.

## Data

The idea behind the baseline is to provided a quick introduction to how to handle the re-id data.
The data files are provided under ```data/synergyreid/raw/synergyreid_data.zip```.

The code extracts the raw files in the same directory and prepares the splits to use for training, validation and test.

Specifically the dataset is divided as:

```shell
SynergyReID dataset loaded
  subset      | # ids | # images
  ---------------------------
  train       |   436 |     8569
  query val   |    50 |      960
  gallery val |    50 |      960
  trainval    |   486 |     9529
  ---------------------------
  query test  |   468 |      468
  gallery test |  8703 |     8703
```

Train and validation identities can be merged (to improve performance) using the flag ```--combine-trainval```.

The validation-set is divided in query and gallery to match the test-set format.
The identities of the gallery are NOT provided; gallery ids are just random.

## Submission

You need to submit a .csv file as the pairwise distance matrix of size m x n (number of query vs number of gallery images).
Please check the correct file order provided by the dataset loader in the baseline ```baseline/synergyreid_baseline.py```.
