# VIPriors object detection challenge

Maintainer: Robert-Jan Bruintjes (r.bruintjes@tudelft.nl)

We present the "Visual Inductive Priors for Data-Efficient Computer Vision" challenge. We offer four challenges, where models are to be trained from scratch, and we reduce the number of training samples to a fraction of the full set. The winners of each challenge are invited to present their winning method at the VIPriors workshop presentation at ECCV 2020.

This challenge is the object detection challenge. We provide a subset of the MS COCO dataset to train on. We will evaluate all models submitted to the challenge on MS COCO validation data.

## Datasets

The task to be performed is object detection, predicting bounding boxes. The training and validation data are subsets of the training split of the MS COCO dataset (2017 release, bounding boxes only). The test set is taken from the validation split of the MS COCO dataset.

As a note: **DO NOT train on MS COCO validation data.** Please use the tooling described here to set up your training, validation and test data to avoid accidentally training on test data.

To find instructions on setting up data please refer to [data README](data/README.md).

## Validation

We provide an evaluation script to test your model over the validation set. Note that this script cannot be used to evaluate models over the testing set, as we do not provide labels for the test set. It is good practice to ensure your predictions work with this script, as the same script is used on the evaluation server.

## Submissions

The evaluation server is hosted using CodaLab. Submitting to the challenge requires a CodaLab account.

~~Please find the evaluation server here~~. *The evaluation server will soon be opened.*

To participate in the challenge one uploads a file of predictions over the challenge test set to the evaluation server. Generate these predictions by inferring your model over our test set (see [data README](data/README.md) for how to get these images) and using the provided `submission_format.py` module to store the predictions. You can also refer to the baseline code in `baseline.py`, which includes example code on how to store prediction results as a submission file.

The submissions file is a JSON encoding of a list of bounding box predictions. This is the format of a submission file:

```json
[
    {
        "image_id": "0.jpg",
        "category_id": 5,
        "bbox": [220,220,30,30],
        "score": 0.98,
    },
    ...
]
```

## Baselines

*Short description of the baseline.*

| **model**           | **checkpoint** | **// METRIC //** |
| ------------------- | -------------- | ---------------- |
| YOUR FANCY BASELINE | SOME PATH      | SOME VALUE       |

*Instructions on adapting the baseline*.