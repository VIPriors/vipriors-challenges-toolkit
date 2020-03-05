"""
Install dependencies using:

```
conda install -c defaults conda-forge --file requirements.txt
```
"""
import os
import json
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import pycocotools

from torchvision_references_detection.engine import perform_eval_inference
from submission_format import save_as_submissions
from evaluate import evaluate as evaluate_submission
from evaluate import evaluate_from_files as evaluate_submission_from_files
from torchvision_references_detection.coco import CocoDetection


DEVICE = 'cuda'
# DEVICE = 'cpu' # TODO: make this automatically use GPU if available
MODEL_NAME = 'vipriors-train-fasterrcnn-resnet50-fpn-0.0005'
LOAD_FROM_EPOCH = 1

if __name__ == '__main__':
    # Init and load trained model (using torchvision.models)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=False, pretrained_backbone=False
    )
    with open(f"{MODEL_NAME}-{LOAD_FROM_EPOCH:04d}.ckpt", 'rb') as f:
        model_data = torch.load(f)
        model.load_state_dict(model_data['model'])

    # Load dataset (using torchvision.datasets.CocoDetection)
    root = 'data/test-images'
    annFile = 'data/annotations/vipriors-object-detection-test.json'

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    dataset = CocoDetection(root, annFile, transform=transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    model.to(DEVICE)
    model.eval()

    #
    # Save predictions
    #

    # Infer model over test set to get predictions
    results = perform_eval_inference(model, loader, DEVICE)

    # Write results to file
    save_as_submissions(results['bbox'], MODEL_NAME)
    print(f"Submission saved as {MODEL_NAME}.json.")



