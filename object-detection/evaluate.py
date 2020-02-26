"""
Use this script to evaluate your model. It stores metrics in the file
`scores.txt`.

Input:
    predictions (str): filepath. Should be a file that matches the submission
        format;
    groundtruths (str): filepath. Should be an annotation file.

Usage:
    evaluate.py <groundtruths> <predictions>
"""
import docopt
import json

import torch
import torchvision
from torchvision_references_detection.engine import evaluate_preloaded


OUTPUT_FILE = 'scores.txt'

def evaluate_from_files(groundtruths_filepath, predictions_filepath):
    """
    Wrapper around evaluation code that reads the objects from JSON files.
    """
    with open(groundtruths_filepath, 'r') as f:
        groundtruths = json.load(f)

    with open(predictions_filepath, 'r') as f:
        results = {'bbox': json.load(f)}

    return evaluate(groundtruths, results)

def evaluate(groundtruths, results):
    """
    Evaluation code
    """
    coco_evaluator = evaluate_preloaded(groundtruths, results)
    mean_ap = coco_evaluator.coco_eval['bbox'].stats[0]

    metrics = [
        ('AP (IoU 0.50-0.95)', mean_ap)
    ]

    # Write metrics to file
    # NOTE(rjbruin): make sure to store metrics as a list of tuples
    # (name (str), value (float))
    # NOTE(rjbruin): `name` cannot contain colons!
    with open(OUTPUT_FILE, 'w') as f:
        for name, val in metrics:
            f.write(f"{name}: {val:.8f}\n")

    print("Metrics written to scores.txt.")

if __name__ == '__main__':
    args = docopt.docopt(__doc__, version='1.0.0')
    evaluate_from_files(args['<groundtruths>'], args['<predictions>'])