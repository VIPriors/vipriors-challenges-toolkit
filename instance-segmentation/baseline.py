"""
Baseline code for the 2021 VIPriors Instance Segmentation challenge based on detectron2.
To use, install detectron2 () and or use the detectron2 default docker image.

Usage:
- Train baseline model:     python baseline.py --dataset_path 'path/to/dataset'
- Predict on test split:    python baseline.py --dataset_path 'path/to/dataset' --predict 'model_weights.pth.tar'

Use the last option for generating the prediction files that can be uploaded to CodaLab.

"""

# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances

# import some common libraries
import torch
import os, argparse, sys, json
from tqdm import tqdm
import numpy as np

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

import pycocotools.mask as mask_util

setup_logger()

def main(args):

    # Register DeepSports dataset
    register_coco_instances("deepsports_train", {}, os.path.join(args.dataset_path,"train.json"),
                            os.path.join(args.dataset_path,"train"))
    register_coco_instances("deepsports_val", {}, os.path.join(args.dataset_path,"val.json"),
                            os.path.join(args.dataset_path,"val"))
    register_coco_instances("deepsports_test", {},
                            os.path.join(args.dataset_path,"test_nolabels.json"),
                            os.path.join(args.dataset_path,"test"))

    # Load config file and adjust where needed
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.DATASETS.TRAIN = ("deepsports_train",)
    cfg.DATASETS.TEST = ("deepsports_val",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = args.bs
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = args.iter
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # human (1), ball (2)
    cfg.MODEL.WEIGHTS = None # transfer learning is prohibited!
    print(cfg)

    # Create output dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if args.predict:
        # load model and weights
        model = build_model(cfg)
        DetectionCheckpointer(model).load(args.predict)
        # make dataloader
        test_loader = build_detection_test_loader(cfg, "deepsports_test")
        # predict
        outputs = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader):
                pred = model(batch)
                pred = batch_to_dict(pred, batch) # convert pred to serializable format
                outputs.extend(pred)

        with open('prediction.json', 'w') as fout:
            json.dump(outputs, fout)

        # exit
        sys.exit()

    # Train
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()
    evaluator = COCOEvaluator("deepsports_val", ("segm",), False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "deepsports_val")
    print(inference_on_dataset(model, val_loader, evaluator))


def batch_to_dict(pred, batch):
    # TODO: check exact format to convert batch into
    results_list = []
    categories = pred[0]['instances'].pred_classes.cpu().numpy().tolist()
    scores = pred[0]['instances'].scores.cpu().numpy().tolist()
    masks = pred[0]['instances'].pred_masks.cpu()

    rles = [mask_util.encode(np.array(mask[:, :, np.newaxis], dtype=np.uint8, order="F"))[0] for mask in masks]
    for rle in rles:
        rle["counts"] = rle["counts"].decode("utf-8")

    for i in range(len(pred[0]['instances'])):
        results_list.append({'image_id': batch[0]['image_id'],
                             'category_id': categories[i]+1,
                             'segmentation': rles[i],
                             'score': scores[i]})

    return results_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VIPriors Segmentation baseline training script')
    parser.add_argument('--dataset_path', metavar='path/to/data/root', default='./data_bytestring',
                        type=str, help='path to dataset (ends with /data)')
    parser.add_argument('--lr', metavar='1e-4', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--bs', metavar='10', default=10, type=int, help='batch size')
    parser.add_argument('--iter', metavar='10000', default=10000, type=int, help='training iterations')
    parser.add_argument('--predict', default=None, type=str, help='provide model weights to run inference')
    args = parser.parse_args()

    print(args)

    main(args)
