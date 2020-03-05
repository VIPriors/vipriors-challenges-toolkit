"""
Use `save_as_submissions()` to save your predictions in a challenge-compatible
file.
"""
import json


def save_as_submissions(model_predictions, filepath):
    """
    Inputs:
        model_predictions (list): list of prediction dictionaries according to
            COCO format:
            ```
            {
                "image_id": int, "category_id": int, "bbox": [x,y,width,height], "score": float,
            }
            ```
        filepath (str): path of the file to be created. The extension `.json`
            will be appended to the filename.
    """
    with open(filepath + '.json', 'w') as f:
        json.dump(model_predictions, f)