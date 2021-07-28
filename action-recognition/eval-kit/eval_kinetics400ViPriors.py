"""
Use this script to evaluate your model. It stores metrics in the file
`scores.txt`.

Input Arguments:
    - -pred (str): filepath. Should be a file that matches the submission
    format.
    - -gt (str): filepath. Should be an annotation file.
    - -outdir (str): dirpath. Should be a path where metric results will be
    stored.
    - -topk (str): List containing the k thresholds for computing the accuracy.

Usage:
    python eval_kinetics400ViPriors.py -pred <filepath> -gt <filepath> -outdir
    <dirpath> -topk <list>
"""
import argparse
import numpy as np
import random
import pandas as pd
from sklearn.metrics import confusion_matrix


def parse_input_args():

    description = ("Evaluation code of the VIPriors Action Recognition "
                   "Challlenge. Please submit your results in the appropriate"
                   " format. Find instructions in the website of the "
                   "challenge.")

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("-pred", help="Prediction file.")
    parser.add_argument("-gt", help="Ground Truth file.")
    parser.add_argument("-outdir", default="./", help="Output directory.")
    parser.add_argument("-topk", default=[1, 3, 5], help="List containing the"
                                                         " K thresholds for"
                                                         " the top-k "
                                                         "accuracy.")

    return parser.parse_args()


def top_k_acc(pred_file, gt_file, k_val, outdir):

    # Read the pred file and the gt file
    with open(pred_file, "r") as pred_f:
        pred = pred_f.readlines()
    
    pred_clips = list()
    pred_cls_1 = list()
    pred_cls_2 = list()
    pred_cls_3 = list()
    pred_cls_4 = list()
    pred_cls_5 = list()
    for line in pred:
        if len(line.split(" ")) >= 2:
            # Get clip name
            pred_clips.append(line.split(" ")[0])

            # Get predicted classes for this clip
            clss = line.split(" ")[1:]
            try:
                pred_cls_1.append(int(clss[0]))
            except:
                pred_cls_1.append(1000)
            try:
                pred_cls_2.append(int(clss[1]))
            except:
                pred_cls_2.append(1000)
            try:
                pred_cls_3.append(int(clss[2]))
            except:
                pred_cls_3.append(1000)
            try:
                pred_cls_4.append(int(clss[3]))
            except:
                pred_cls_4.append(1000)
            try:
                pred_cls_5.append(int(clss[4]))
            except:
                pred_cls_5.append(1000)
        else:
            # No predicted labels found for this clip
            pred_cls_1.append(1000)
            pred_cls_2.append(1000)
            pred_cls_3.append(1000)
            pred_cls_4.append(1000)
            pred_cls_5.append(1000)
    
    # DataFrame with predictions
    preds_df = pd.DataFrame({"videoid": pred_clips,
                             "cls1": pred_cls_1,
                             "cls2": pred_cls_2,
                             "cls3": pred_cls_3,
                             "cls4": pred_cls_4,
                             "cls5": pred_cls_5,
                             })

    # Read gt
    gt_df = pd.read_csv(gt_file)
    
    # Read class index
    clsidx_file = "../data/annotations/clsIdx.csv"
    clsidx_df = pd.read_csv(clsidx_file)

    # Check predictions
    groundtruth = np.zeros((1, len(gt_df)))
    predictions = np.zeros((len(k_val), len(gt_df)))
    for index, row in gt_df.iterrows():
        # Get gt class idx
        gt_cls_df = clsidx_df.loc[clsidx_df["label"] == row.label]
        gt_cls_idx = gt_cls_df["idx"].values[0]
        groundtruth[0, index] = gt_cls_idx

        # Get pred classes idx for each top k value
        this_pred_df = preds_df.loc[preds_df["videoid"] == row.videoid]
        if this_pred_df.empty == True:
            print(f"No predictions found for video id {row.videoid}.")
            for i, topk in enumerate(k_val):
                while True:
                    random_cls = random.randint(1, 400)
                    if random_cls != gt_cls_idx:
                        predictions[i, index] = random_cls
                        break
        else:
            this_pred_cls_df = this_pred_df.drop(["videoid"], axis=1).values
            this_pred_cls = this_pred_cls_df[0].tolist()
            for i, topk in enumerate(k_val):
                if gt_cls_idx in this_pred_cls[:topk]:
                    predictions[i, index] = gt_cls_idx
                else:
                    if this_pred_cls[(topk - 1)] in list(range(1, 400, 1)):
                        predictions[i, index] = this_pred_cls[(topk - 1)]
                    else:
                        while True:
                            random_cls = random.randint(1, 400)
                            if random_cls != gt_cls_idx:
                                predictions[i, index] = random_cls
                                break
                    
    # Get Accuracy per class all at once with the confusion matrix
    topkacc = np.zeros((1, len(k_val)))
    for k, topk in enumerate(k_val):
        labels = list(range(1, 400, 1))
        # Confusion matrix
        cf = confusion_matrix(groundtruth[0, :], predictions[k, :],
                              labels=labels).astype(float)

        # Class accuracy by averaging diagonal. Mean Avg Acc, averaging evrthng
        n_per_cls = cf.sum(axis=1)
        hits_per_cls = np.diag(cf)
        topkacc[0, k] = np.mean(hits_per_cls / n_per_cls)

    # Save scores to file
    with open(f'{outdir}scores.txt', 'w') as score_f:
        for l, topk in enumerate(k_val):
            score_f.write(f'Top-{topk} accuracy: {topkacc[0, l]}\n')

    return


if __name__ == '__main__':
    # Get input arguments
    pred_file, gt_file, outdir, k = vars(parse_input_args()).values()

    # Compute accuracy
    print('Calculating Accuracy...')
    top_k_acc(pred_file, gt_file, k, outdir)
    print('DONE!')

