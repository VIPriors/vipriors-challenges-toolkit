r"""The script to generate submission file, please use fake_test_annotations.json
file.
Test submission:
python generate_submission.py  --data_path </data/DelftBikes/> \
 --test_json <fake_test_annotations.json> --resume <checkpoint>
Val submission:
python generate_submission.py  --data_path </data/DelftBikes/> \
--eval_mode val --test_json <val_annotations.json> --resume <checkpoint>
 
Please compress submission.json file as submission.zip without containing any other 
files or folders before submiting to the server.
"""
import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
import torchvision
import torchvision.models.detection
from baseline.engine import perform_eval_inference
from baseline.utils import save_as_submissions
import baseline.utils as utils
from baseline.models import models
from dataset.bike_dataset import DelftBikeDataset
from dataset.dataset_utils import get_transform

def main(args):
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")

    test_set = DelftBikeDataset(args.data_path, args.test_json,
                                get_transform(train=False), mode=args.eval_mode)
    print('test_set', len(test_set))

    data_loader_test = torch.utils.data.DataLoader(
            test_set, batch_size=1, shuffle=False, num_workers=2,
            collate_fn=utils.collate_fn)

    print("Creating model")
    model = models[args.model](num_classes=23)  # 22 parts + 1 Background
    model.to(device)

    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model'])

    # Infer model over test set to get predictions
    results = perform_eval_inference(model, data_loader_test, device)

    # Write results to file
    save_as_submissions(results['bbox'], args.file)
    print(f"Submission saved as {args.file}.json.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--data_path', default='/data/DelftBikes',
                        help='dataset')
    parser.add_argument('--eval_mode', default='test',
                        help='set to evaluate, (val or test)')
    parser.add_argument('--test_json', default='fake_test_annotations.json',
                        help='fake testing labels')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)

    parser.add_argument('--file', default='submission',
                        help='Filename for submission file (without file extension).')

    args = parser.parse_args()

    main(args)
