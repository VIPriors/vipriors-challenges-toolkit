# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 13:37:37 2019 by Attila Lengyel - attila@lengyel.nl

Write this

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF

import time
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import warnings
import random
from PIL import Image

from helpers.model import UNet
from helpers.minicity import MiniCity
from helpers.helpers import AverageMeter, ProgressMeter, iouCalc

parser = argparse.ArgumentParser(description='VIPriors Segmentation baseline training script')
parser.add_argument('--dataset_path', metavar='path/to/minicity/root', default='./minicity',
                    type=str, help='path to dataset (ends with /minicity)')
parser.add_argument('--colorjitter_factor', metavar='0.3', default=0.3,
                    type=float, help='data augmentation: color jitter factor')
parser.add_argument('--scale_factor', metavar='0.3', default=0.3,
                    type=float, help='data augmentation: random scale factor')
parser.add_argument('--hflip', metavar='[True,False]', default=True,
                    type=float, help='data augmentation: random horizontal flip')
parser.add_argument('--crop_size', metavar='384 768', default=[384,768], nargs="+",
                    type=int, help='data augmentation: random crop size, height width, space separated')
parser.add_argument('--train_size', metavar='512 1024', default=[512,1024], nargs="+",
                    type=int, help='image size during training, height width, space separated')
parser.add_argument('--test_size', metavar='512 1024', default=[512,1024], nargs="+",
                    type=int, help='image size during validation and testing, height width, space separated')
parser.add_argument('--batch_size', metavar='5', default=5, type=int, help='batch size')
parser.add_argument('--pin_memory', metavar='[True,False]', default=True,
                    type=bool, help='pin memory on GPU')
parser.add_argument('--num_workers', metavar='8', default=8, type=int,
                    help='number of dataloader workers')
parser.add_argument('--lr_init', metavar='1e-2', default=1e-2, type=float,
                    help='initial learning rate')
parser.add_argument('--lr_min', metavar='1e-5', default=1e-5, type=float,
                    help='lower bound on learning rate')
parser.add_argument('--lr_patience', metavar='5', default=5, type=int,
                    help='patience for reduce learning rate on plateau')
parser.add_argument('--lr_momentum', metavar='0.9', default=0.9, type=float,
                    help='momentum for SGD optimizer')
parser.add_argument('--lr_weight_decay', metavar='1e-4', default=1e-4, type=float,
                    help='weight decay for SGD optimizer')
parser.add_argument('--weights', metavar='path/to/checkpoint', default=None,
                    type=str, help='resume training from checkpoint')
parser.add_argument('--epochs', metavar='200', default=200, type=int,
                    help='number of training epochs')
parser.add_argument('--seed', metavar='42', default=None, type=int,
                    help='random seed to use')
parser.add_argument('--dataset_mean', metavar='[0.485, 0.456, 0.406]',
                    default=[0.485, 0.456, 0.406], type=list,
                    help='mean for normalization')
parser.add_argument('--dataset_std', metavar='[0.229, 0.224, 0.225]',
                    default=[0.229, 0.224, 0.225], type=list,
                    help='std for normalization')
parser.add_argument('--predict', metavar='path/to/weights',
                    default=None, type=str,
                    help='provide path to model weights to predict on validation set')

"""
===========
Main method
===========
"""

def main(): 
    global args
    args = parser.parse_args()
    args.crop_size = tuple(args.crop_size)
    args.train_size = tuple(args.train_size)
    args.test_size = tuple(args.test_size)

    # Fix seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    assert args.crop_size[0] <= args.train_size[0] and args.crop_size[1] <= args.train_size[1], \
    'Crop size must be <= image size.'
    
    # Create directory to store run files
    if not os.path.isdir('baseline_run'):
        os.makedirs('baseline_run/images')
        os.makedirs('baseline_run/results_color')
    
    # Load dataset
    trainset = MiniCity(args.dataset_path, split='train', transforms=train_trans)
    valset = MiniCity(args.dataset_path, split='val', transforms=test_trans)
    testset = MiniCity(args.dataset_path, split='test', transforms=test_trans)
    dataloaders = {}    
    dataloaders['train'] = torch.utils.data.DataLoader(trainset,
               batch_size=args.batch_size, shuffle=True,
               pin_memory=args.pin_memory, num_workers=args.num_workers)
    dataloaders['val'] = torch.utils.data.DataLoader(valset,
               batch_size=args.batch_size, shuffle=False,
               pin_memory=args.pin_memory, num_workers=args.num_workers)
    dataloaders['test'] = torch.utils.data.DataLoader(testset,
               batch_size=args.batch_size, shuffle=False,
               pin_memory=args.pin_memory, num_workers=args.num_workers)
    
    # Load model
    model = UNet(len(MiniCity.validClasses), batchnorm=True)
    
    # Define loss, optimizer and scheduler
    criterion = nn.CrossEntropyLoss(ignore_index=MiniCity.voidClass)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_init,
                                momentum=args.lr_momentum,
                                weight_decay=args.lr_weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        patience=args.lr_patience, min_lr=args.lr_min)
    
    # Initialize metrics
    best_miou = 0.0
    metrics = {'train_loss' : [],
               'train_acc' : [],
               'val_acc' : [],
               'val_loss' : [],
               'miou' : []}
    start_epoch = 0

    # Push model to GPU
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        print('Model pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))
    
    # Resume training from checkpoint
    if args.weights:
        print('Resuming training from {}.'.format(args.weights))
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        metrics = checkpoint['metrics']
        best_miou = checkpoint['best_miou']
        start_epoch = checkpoint['epoch']+1

    # No training, only running prediction on test set
    if args.predict:
        checkpoint = torch.load(args.predict)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print('Loaded model weights from {}'.format(args.predict))
        # Create results directory
        if not os.path.isdir('results'):
            os.makedirs('results')
        predict(dataloaders['test'], model, MiniCity.mask_colors)
        return
    
    # Generate log file
    with open('baseline_run/log_epoch.csv', 'a') as epoch_log:
        epoch_log.write('epoch, train loss, val loss, train acc, val acc, miou\n')
    
    since = time.time()
    
    for epoch in range(start_epoch,args.epochs):
        
        # Train
        print('--- Training ---')
        train_loss, train_acc = train_epoch(dataloaders['train'], model,
                                            criterion, optimizer, scheduler,
                                            epoch, void=MiniCity.voidClass)
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        print('Epoch {} train loss: {:.4f}, acc: {:.4f}'.format(epoch,train_loss,train_acc))
        
        # Validate
        print('--- Validation ---')
        val_acc, val_loss, miou = validate_epoch(dataloaders['val'],
                                                 model,
                                                 criterion, epoch,
                                                 MiniCity.classLabels,
                                                 MiniCity.validClasses,
                                                 void=MiniCity.voidClass,
                                                 maskColors=MiniCity.mask_colors)
        metrics['val_acc'].append(val_acc)
        metrics['val_loss'].append(val_loss)
        metrics['miou'].append(miou)
        
        # Write logs
        with open('baseline_run/log_epoch.csv', 'a') as epoch_log:
            epoch_log.write('{}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}\n'.format(
                    epoch, train_loss, val_loss, train_acc, val_acc, miou))
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_miou': best_miou,
            'metrics': metrics,
            }, 'baseline_run/checkpoint.pth.tar')
        
        # Save best model to file
        if miou > best_miou:
            print('mIoU improved from {:.4f} to {:.4f}.'.format(best_miou, miou))
            best_miou = miou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                }, 'baseline_run/best_weights.pth.tar')
                
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    # Plot learning curves
    x = np.arange(args.epochs)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ln1 = ax1.plot(x, metrics['train_loss'], color='tab:red')
    ln2 = ax1.plot(x, metrics['val_loss'], color='tab:red', linestyle='dashed')
    ax1.grid()
    ax2 = ax1.twinx()
    ax2.set_ylabel('accuracy')
    ln3 = ax2.plot(x, metrics['train_acc'], color='tab:blue')
    ln4 = ax2.plot(x, metrics['val_acc'], color='tab:blue', linestyle='dashed')
    ln5 = ax2.plot(x, metrics['miou'], color='tab:green')
    lns = ln1+ln2+ln3+ln4+ln5
    plt.legend(lns, ['Train loss','Validation loss','Train accuracy','Validation accuracy','mIoU'])
    plt.tight_layout()
    plt.savefig('baseline_run/learning_curve.pdf', bbox_inches='tight')
    
    # Load best model
    checkpoint = torch.load('baseline_run/best_weights.pth.tar')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print('Loaded best model weights (epoch {}) from baseline_run/best_weights.pth.tar'.format(checkpoint['epoch']))
    # Create results directory
    if not os.path.isdir('results'):
        os.makedirs('results')
    # Run prediction on validation set
    # For predicting on test set, simple replace 'val' by 'test'
    predict(dataloaders['val'], model, MiniCity.mask_colors)

"""
=================
Routine functions
=================
"""

def train_epoch(dataloader, model, criterion, optimizer, lr_scheduler, epoch, void=-1):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_running = AverageMeter('Loss', ':.4e')
    acc_running = AverageMeter('Accuracy', ':.3f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, loss_running, acc_running],
        prefix="Train, epoch: [{}]".format(epoch))
    
    # input resolution
    if args.crop_size is not None:
        res = args.crop_size[0]*args.crop_size[1]
    else:
        res = args.train_size[0]*args.train_size[1]
    
    # Set model in training mode
    model.train()
    
    end = time.time()
    
    with torch.set_grad_enabled(True):
        # Iterate over data.
        for epoch_step, (inputs, labels, _) in enumerate(dataloader):
            data_time.update(time.time()-end)
            
            inputs = inputs.float().cuda()
            labels = labels.long().cuda()
            
            # zero the parameter gradients
            optimizer.zero_grad()
        
            # forward pass
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            loss = criterion(outputs, labels)
            
            # backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            bs = inputs.size(0) # current batch size
            loss = loss.item()
            loss_running.update(loss, bs)
            corrects = torch.sum(preds == labels.data)
            nvoid = int((labels==void).sum())
            acc = corrects.double()/(bs*res-nvoid) # correct/(batch_size*resolution-voids)
            acc_running.update(acc, bs)
            
            # output training info
            progress.display(epoch_step)
            
            # Measure time
            batch_time.update(time.time() - end)
            end = time.time()

        # Reduce learning rate
        lr_scheduler.step(loss_running.avg)
        
    return loss_running.avg, acc_running.avg

    
def validate_epoch(dataloader, model, criterion, epoch, classLabels, validClasses, void=-1, maskColors=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_running = AverageMeter('Loss', ':.4e')
    acc_running = AverageMeter('Accuracy', ':.4e')
    iou = iouCalc(classLabels, validClasses, voidClass = void)
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, loss_running, acc_running],
        prefix="Test, epoch: [{}]".format(epoch))
    
    # input resolution
    res = args.test_size[0]*args.test_size[1]
    
    # Set model in evaluation mode
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        for epoch_step, (inputs, labels, filepath) in enumerate(dataloader):
            data_time.update(time.time()-end)
            
            inputs = inputs.float().cuda()
            labels = labels.long().cuda()
    
            # forward
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Statistics
            bs = inputs.size(0) # current batch size
            loss = loss.item()
            loss_running.update(loss, bs)
            corrects = torch.sum(preds == labels.data)
            nvoid = int((labels==void).sum())
            acc = corrects.double()/(bs*res-nvoid) # correct/(batch_size*resolution-voids)
            acc_running.update(acc, bs)
            # Calculate IoU scores of current batch
            iou.evaluateBatch(preds, labels)
            
            # Save visualizations of first batch
            if epoch_step == 0 and maskColors is not None:
                for i in range(inputs.size(0)):
                    filename = os.path.splitext(os.path.basename(filepath[i]))[0]
                    # Only save inputs and labels once
                    if epoch == 0:
                        img = visim(inputs[i,:,:,:])
                        label = vislbl(labels[i,:,:], maskColors)
                        if len(img.shape) == 3:
                            cv2.imwrite('baseline_run/images/{}.png'.format(filename),img[:,:,::-1])
                        else: 
                            cv2.imwrite('baseline_run/images/{}.png'.format(filename),img)
                        cv2.imwrite('baseline_run/images/{}_gt.png'.format(filename),label[:,:,::-1])
                    # Save predictions
                    pred = vislbl(preds[i,:,:], maskColors)
                    cv2.imwrite('baseline_run/images/{}_epoch_{}.png'.format(filename,epoch),pred[:,:,::-1])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # print progress info
            progress.display(epoch_step)
        
        miou = iou.outputScores()
        print('Accuracy      : {:5.3f}'.format(acc_running.avg))
        print('---------------------')

    return acc_running.avg, loss_running.avg, miou

def predict(dataloader, model, maskColors):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time],
        prefix='Predict: ')
    
    # Set model in evaluation mode
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        for epoch_step, batch in enumerate(dataloader):

            if len(batch) == 2:
                inputs, filepath = batch
            else:
                inputs, _, filepath = batch

            data_time.update(time.time()-end)
            
            inputs = inputs.float().cuda()
    
            # forward
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            
            # Save visualizations of first batch
            for i in range(inputs.size(0)):
                filename = os.path.splitext(os.path.basename(filepath[i]))[0]
                # Save input
                img = visim(inputs[i,:,:,:])
                img = Image.fromarray(img, 'RGB')
                img.save('baseline_run/results_color/{}_input.png'.format(filename))
                # Save prediction with color labels
                pred = preds[i,:,:].cpu()
                pred_color = vislbl(pred, maskColors)
                pred_color = Image.fromarray(pred_color.astype('uint8'))
                pred_color.save('baseline_run/results_color/{}_prediction.png'.format(filename))
                # Save class id prediction (used for evaluation)
                pred_id = MiniCity.trainid2id[pred]
                pred_id = Image.fromarray(pred_id)
                pred_id = pred_id.resize((2048,1024), resample=Image.NEAREST)
                pred_id.save('results/{}.png'.format(filename))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # print progress info
            progress.display(epoch_step)

"""
====================
Data transformations
====================
"""

def test_trans(image, mask=None):
    # Resize, 1 for Image.LANCZOS
    image = TF.resize(image, args.test_size, interpolation=1) 
    # From PIL to Tensor
    image = TF.to_tensor(image)
    # Normalize
    image = TF.normalize(image, args.dataset_mean, args.dataset_std)
    
    if mask:
        # Resize, 0 for Image.NEAREST
        mask = TF.resize(mask, args.test_size, interpolation=0) 
        mask = np.array(mask, np.uint8) # PIL Image to numpy array
        mask = torch.from_numpy(mask) # Numpy array to tensor
        return image, mask
    else:
        return image

def train_trans(image, mask):
    # Generate random parameters for augmentation
    bf = np.random.uniform(1-args.colorjitter_factor,1+args.colorjitter_factor)
    cf = np.random.uniform(1-args.colorjitter_factor,1+args.colorjitter_factor)
    sf = np.random.uniform(1-args.colorjitter_factor,1+args.colorjitter_factor)
    hf = np.random.uniform(-args.colorjitter_factor,+args.colorjitter_factor)
    scale_factor = np.random.uniform(1-args.scale_factor,1+args.scale_factor)
    pflip = np.random.randint(0,1) > 0.5

    # Resize, 1 for Image.LANCZOS
    image = TF.resize(image, args.train_size, interpolation=1)
    # Resize, 0 for Image.NEAREST
    mask = TF.resize(mask, args.train_size, interpolation=0) 
    
    # Random scaling
    image = TF.affine(image, 0, [0,0], scale_factor, [0,0])
    mask = TF.affine(mask, 0, [0,0], scale_factor, [0,0])

    # Random cropping
    if not args.train_size == args.crop_size:
        # From PIL to Tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        h, w = args.train_size
        th, tw = args.crop_size
        i = np.random.randint(0, h - th)
        j = np.random.randint(0, w - tw)
        image = image[:,i:i+th,j:j+tw]
        mask = mask[:,i:i+th,j:j+tw]
        image = TF.to_pil_image(image)
        mask = TF.to_pil_image(mask[0,:,:])
    
    # H-flip
    if pflip == True and args.hflip == True:
        image = TF.hflip(image)
        mask = TF.hflip(mask)
    
    # Color jitter
    image = TF.adjust_brightness(image, bf)
    image = TF.adjust_contrast(image, cf)
    image = TF.adjust_saturation(image, sf)
    image = TF.adjust_hue(image, hf)

    # From PIL to Tensor
    image = TF.to_tensor(image)
    
    # Normalize
    image = TF.normalize(image, args.dataset_mean, args.dataset_std)
    
    # Convert ids to train_ids
    mask = np.array(mask, np.uint8) # PIL Image to numpy array
    mask = torch.from_numpy(mask) # Numpy array to tensor
        
    return image, mask

"""
================
Visualize images
================
"""

def visim(img):
    img = img.cpu()
    # Convert image data to visual representation
    img *= torch.tensor(args.dataset_std)[:,None,None]
    img += torch.tensor(args.dataset_mean)[:,None,None]
    npimg = (img.numpy()*255).astype('uint8')
    if len(npimg.shape) == 3 and npimg.shape[0] == 3:
        npimg = np.transpose(npimg, (1, 2, 0))
    else:
        npimg = npimg[0,:,:]
    return npimg

def vislbl(label, mask_colors):
    label = label.cpu()
    # Convert label data to visual representation
    label = np.array(label.numpy())
    if label.shape[-1] == 1:
        label = label[:,:,0]
    
    # Convert train_ids to colors
    label = mask_colors[label]
    return label
    
if __name__ == '__main__':
    main()