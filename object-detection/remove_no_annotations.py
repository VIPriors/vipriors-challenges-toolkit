"""
Remove degenerate annotations (non-positive height or width) and images without
annotations from an annotation file.

Usage:
    remove_no_annotations.py <annotation_file> <new_file>
"""
import json
import docopt
import tqdm


def clean(annotations_file, new_file):
    with open(annotations_file, 'r') as f:
        data = json.load(f)

    # Find all images that have annotations
    image_id_has_annotations = {}
    clean_annotations = []
    skipped_annotations = 0
    for ann_obj in tqdm.tqdm(data['annotations'], desc="Remove degenenate annotations"):
        # Check bbox
        _, _, w, h = ann_obj['bbox']
        if w <= 0. or h <= 0.:
            skipped_annotations += 1
            # Skip degenerate annotations, so we remove them from dataset
            continue

        image_id = ann_obj['image_id']
        image_id_has_annotations[image_id] = True

        clean_annotations.append(ann_obj)

    data['annotations'] = clean_annotations
    print(f"Removed {skipped_annotations} degenerate annotation(s).")

    # Filter images by having annotations
    new_images = []
    skipped_images = 0
    for im_obj in tqdm.tqdm(data['images'], desc="Remove images without annotations"):
        im_id = im_obj['id']
        if im_id in image_id_has_annotations:
            new_images.append(im_obj)
        else:
            skipped_images += 1

    print(f"Removed {skipped_images} image(s) without annotations.")

    # Overwrite images and save data
    print("Write new annotations to file...")
    data['images'] = new_images
    with open(new_file, 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    clean(args['<annotation_file>'], args['<new_file>'])