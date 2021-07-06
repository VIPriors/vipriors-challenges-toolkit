"""
Use this script to evaluate your model. It stores metrics in the file
`scores.txt`.

Input:
    predictions (str): filepath. Should be a file that matches the submission
        format;
    groundtruths (str): filepath. Should be an annotation file.
    output dir (str): filepath. Existing directory where .json should be stored.

Usage:
    evaluate.py <groundtruths> <predictions> <output_dir>
"""
import json
import os
import sys
import time

from pycocotools.coco import COCO
from coco_eval import CocoEvaluator


OUTPUT_FILE = 'scores.txt'

def evaluate_preloaded(gt_from_file, results_from_file):
    # Use dataset object loaded from file instead of from dataset
    coco = COCO()
    coco.dataset = gt_from_file
    coco.createIndex()

    iou_types = ["segm"]
    coco_evaluator = CocoEvaluator(coco, iou_types)
    coco_evaluator.put_results(results_from_file)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    # torch.set_num_threads(torch.get_num_threads())
    return coco_evaluator

def evaluate_from_files(groundtruths_filepath, predictions_filepath, output_dir):
    """
    Wrapper around evaluation code that reads the objects from JSON files.
    """
    with open(groundtruths_filepath, 'r') as f:
        groundtruths = json.load(f)

    with open(predictions_filepath, 'r') as f:
        results = {'segm': json.load(f)}

    return evaluate(groundtruths, results, output_dir)

def evaluate(groundtruths, results, output_dir):
    """
    Evaluation code
    """
    start = time.time()
    coco_evaluator = evaluate_preloaded(groundtruths, results)

    all_stats = coco_evaluator.coco_eval['segm'].stats
    metrics = [
        ("AP @ 0.50-0.95", all_stats[0]),
        ("AP @ 0.50", all_stats[1]),
        ("AP @ 0.75", all_stats[2]),
        ("AP @ 0.50-0.95 (small)", all_stats[3]),
        ("AP @ 0.50-0.95 (medium)", all_stats[4]),
        ("AP @ 0.50-0.95 (large)", all_stats[5])
    ]

    # Write metrics to file
    # NOTE(rjbruin): make sure to store metrics as a list of tuples
    # (name (str), value (float))
    # NOTE(rjbruin): `name` cannot contain colons!
    with open(os.path.join(output_dir, OUTPUT_FILE), 'w') as f:
        for name, val in metrics:
            f.write(f"{name}: {val:.8f}\n")

    print("Metrics written to scores.txt.")

if __name__ == '__main__':
    args = sys.argv[1:]
    evaluate_from_files(args[0], args[1], args[2])
