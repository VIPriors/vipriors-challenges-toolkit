r"""PyTorch Detection Training on DelftBikes dataset.

The dataset contains 10,000 images with 22 bikes parts. Each part has 
class label, bounding box locations and state labels that indicates being intact,
absent, damaged and occluded.
During evaluation, only intact, damaged and occluded parts will be used for evaluation.

We provide train labels and fake test labels to be able to generate submission. 
If you want, you can use valset_generate.py script to derive valse from train set.
Valset can be used to validate your submissions.

Image sizes are kept as original sizes and evaluation is also done on original sizes.
Baseline model does not have any data augmentation. 
The default hyperparameters are tuned for training on a single GPU and with 
batch size of 4.

To train the Faster RCNN network,
 python train_baseline.py --data_path </data/DelftBikes/> \
 --train_json <train_annotations.json>
 
To train the Faster RCNN network with valset and new train set.
First, use use valset_generate.py. Then,
 python train_baseline.py --data_path </data/DelftBikes/> \
 --train_json <new_rain_annotations.json> --eval_mode val \
 --test_json <val_annotations.json>
"""
import datetime
import os
import time
import torch
import torch.utils.data
from torch import nn
import torchvision
import torchvision.models.detection
from baseline.engine import train_one_epoch, evaluate
import baseline.utils as utils
from baseline.models import models
from dataset.bike_dataset import DelftBikeDataset
from dataset.dataset_utils import get_transform

def main(args):
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")

    train_set = DelftBikeDataset(
            args.data_path, args.train_json, get_transform(train=False), mode='train')
    test_set = DelftBikeDataset(
            args.data_path, args.test_json, get_transform(train=False), mode=args.eval_mode)
    print('train_set', len(train_set))
    print('test_set', len(test_set))

    data_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
            collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False, num_workers=4,
            collate_fn=utils.collate_fn)

    print("Creating model")
    model = models[args.model](num_classes=23)  # 22 parts + 1 Background
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        evaluate(model, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()
        if args.output_dir:
            utils.save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch},
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

        # evaluate after every epoch
        evaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--data_path', default='/data/DelftBikes',
                        help='dataset')
    parser.add_argument('--eval_mode', default='test',
                        help='set to evaluate, (val or test)')
    parser.add_argument('--train_json', default='train_annotations.json', help='train labels')
    parser.add_argument('--test_json', default='fake_test_annotations.json', help='fake testing labels')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=16, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                        'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-steps', default=[15, 30], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=250, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true", default=False)

    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)
