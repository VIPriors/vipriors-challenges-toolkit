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
    python eval_moducf101.py -pred <filepath> -gt <filepath> -outdir
    <dirpath> -topk <list>
"""
import argparse
import numpy as np
import random
from sklearn.metrics import confusion_matrix


def parse_input_args():

    description = ('Evaluation code of the VIPriors Action Recognition '
                   'Challlenge. Please submit your results in the appropriate '
                   'format. Find instructions in the website of the challenge.')

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-pred', help='Prediction file.')
    parser.add_argument('-gt', help='Ground Truth file.')
    parser.add_argument('-outdir', default='./', help='Output directory.')
    parser.add_argument('-topk', default=[1, 3, 5], help='List containing the K'
                                                         ' thresholds for the '
                                                         'top-k accuracy.')

    return parser.parse_args()


def top_k_acc(pred_file, gt_file, k_val, outdir):

    # Read the pred file and the gt file
    with open(pred_file, 'r') as pred_f:
        pred = pred_f.readlines()
        pred_clips = list()
        pred_cls = list()
        for line in pred:
            if len(line.split(' ')) >= 2:
                pred_clips.append(line.split(' ')[0])
                clss = line.split(' ')[1:]
                clss_lst = list()
                for c in clss:
                    clss_lst.append(int(c))
            else:
                # No label predicted found
                clss_lst = list()
                clss_lst.append(0)
            pred_cls.append(clss_lst)
    with open(gt_file, 'r') as gt_f:
        gt = gt_f.readlines()
        gt_clips = list()
        gt_cls = list()
        for line in gt:
            gt_clips.append(line.split(' ')[0])
            gt_cls.append(int(line.split(' ')[1]))

    # Check predictions
    groundtruth = np.zeros((1, len(gt)))
    predictions = np.zeros((len(k_val), len(gt)))
    for i, clip in enumerate(gt_clips):
        # Get ground truth label
        groundtruth[0, i] = gt_cls[i]

        # Check if video is in the submitted file and get the index of it
        try:
            pred_idx = pred_clips.index(clip)
        except:
            # If it's not error. Assigning random label not the gt one
            random_cls = random.randint(1, 101)
            if gt_cls[i] == random_cls:
                if random_cls == 1:
                    random_cls = random_cls + 1
                elif random_cls == 101:
                    random_cls = random_cls - 1
            predictions[:, i] = random_cls
            continue

        # Get predicted labels for this clip
        this_clip_pred = pred_cls[pred_idx]

        # Check if prediction is in top-k
        for j, topk in enumerate(k_val):
            if gt_cls[i] in this_clip_pred[:topk]:
                predictions[j, i] = gt_cls[i]
            else:
                if this_clip_pred[0] == 0:
                    # Was empty label. Assigning random label not the gt one
                    random_cls = random.randint(1, 101)
                    if gt_cls[i] == random_cls:
                        if random_cls == 1:
                            random_cls = random_cls + 1
                        elif random_cls == 101:
                            random_cls = random_cls - 1
                    predictions[j, i] = random_cls
                else:
                    predictions[j, i] = this_clip_pred[0]

    # Get Accuracy per class all at once with the confusion matrix
    topkacc = np.zeros((1, len(k_val)))
    for k, topk in enumerate(k_val):
        labels = list(range(1, 102, 1))
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
