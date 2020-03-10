r"""To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU
"""
import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn

from baseline.coco_utils import get_coco, get_coco_kp
from baseline.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from baseline.engine import perform_eval_inference
import baseline.utils as utils
import baseline.transforms as T
from baseline.models import models
from submission_format import save_as_submissions


def get_dataset(name, image_set, transform, data_path):
    paths = {
        "coco": (data_path, get_coco, 91),
        "coco_kp": (data_path, get_coco_kp, 2)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")

    # NOTE(rjbruin): Modified to use our annotation files
    # dataset, num_classes = get_dataset(args.dataset, "train", get_transform(train=True), args.data_path)
    dataset_test, num_classes = get_dataset(args.dataset, args.split, get_transform(train=False), args.data_path)

    print("Creating data loaders")
    if args.distributed:
        # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        # train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    # if args.aspect_ratio_group_factor >= 0:
        # group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        # train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    # else:
        # train_batch_sampler = torch.utils.data.BatchSampler(
            # train_sampler, args.batch_size, drop_last=True)

    # data_loader = torch.utils.data.DataLoader(
    #     dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
    #     collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    # NOTE: Changed model to a custom defined Faster R-CNN with smaller ResNet
    print("Creating model")
    model = models[args.model](num_classes=num_classes)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(
    #     params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    # if args.resume:
    checkpoint = torch.load(args.resume, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # args.start_epoch = checkpoint['epoch'] + 1

    # Infer model over test set to get predictions
    results = perform_eval_inference(model, data_loader_test, device)

    # Write results to file
    save_as_submissions(results['bbox'], args.file)
    print(f"Submission saved as {args.file}.json.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--data-path', default='/datasets01/COCO/022719/', help='dataset')
    parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--model', default='maskrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1', help='url used to set up distributed training')
    parser.add_argument('--dist-port', default='12345')

    # CUSTOM
    parser.add_argument('--split', default='test', help='Dataset split to generate predictions for.')
    parser.add_argument('--file', default='submission', help='Filename for submission file (without file extension).')

    args = parser.parse_args()

    main(args)
