"""
Install dependencies using:

```
conda install -c defaults conda-forge --file requirements.txt
```

Usage:
    baseline.py <learning_rate>
"""
import os
import json
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import pycocotools
import docopt

from torchvision_references_detection.engine import train_one_epoch, evaluate
import torchvision_references_detection.utils as utils
from torchvision_references_detection.coco import CocoDetection


DEVICE = 'cuda'
MODEL_NAME = "vipriors-train-fasterrcnn-resnet50-fpn-{lr}"
EPOCHS = 4

# DEBUG
# torch.autograd.set_detect_anomaly(True)
torch.autograd.set_detect_anomaly(False)
print("Anomaly detection enabled: ", torch.is_anomaly_enabled())

if __name__ == '__main__':
    args = docopt.docopt(__doc__)

    learning_rate = float(args['<learning_rate>'])
    model_name = MODEL_NAME.format(lr=learning_rate)

    # Create model (using torchvision.models). NOTE: no pre-training!
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=False, pretrained_backbone=False
    )

    # Load dataset (using torchvision.datasets.CocoDetection)
    # root = '/data/VisionLab/vipriors-object-detection/images'
    root = 'data/images'
    annFile = 'data/annotations/vipriors-object-detection-train.json'
    annFileTest = 'data/annotations/vipriors-object-detection-val.json'

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    dataset = CocoDetection(root, annFile, transform=transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True,
                                         num_workers=4, collate_fn=utils.collate_and_make_tensors)

    test_dataset = CocoDetection(root, annFileTest, allow_missing_targets=False, transform=transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                              num_workers=4, collate_fn=utils.collate_and_make_tensors)

    model.to(DEVICE)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    for epoch in range(EPOCHS):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, loader, DEVICE, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # save model
        model_save_path = f"{model_name}-{epoch:04d}.ckpt"
        print(f"saving model at {model_save_path}")
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'lr_scheduler': lr_scheduler.state_dict(),
        }, model_save_path)
        # evaluate on the test dataset
        evaluate(model, test_loader, device=DEVICE)



