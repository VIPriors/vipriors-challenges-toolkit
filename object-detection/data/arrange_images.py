"""
Arrange training, validation and test data for VIPriors object detection
challenge.
"""
import tqdm
import os
import shutil


if __name__ == '__main__':
    if os.path.exists("TEST_MARKER_DO_NOT_REMOVE"):
        raise ValueError("Test set has already been updated. Don't run this "
                         "script again!")

    if os.path.exists("test-images"):
        raise ValueError("Test set writing directory exists. Please make sure "
                         "the writing directory does not exist, as the "
                         "resulting folder must be empty except for the newly "
                         "written test set.")
    os.makedirs("test-images")

    print("Loading VIPriors testing split settings...")
    with open('annotations/test_image_mappings.txt', 'r') as f:
        lines = f.read().split("\n")
        test_mappings = {}
        for line in lines:
            key, val = line.strip().split(",")
            test_mappings[int(key)] = int(val)

    for old_id, new_id in tqdm.tqdm(test_mappings.items(), desc='Processing images'):
        shutil.copyfile(f"val2017/{old_id:012d}.jpg", f"test-images/{new_id:012d}.jpg")

    # Add a marker file to the folder to indicate that test data was updated
    with open("TEST_MARKER_DO_NOT_REMOVE", 'w') as f:
        f.write("This file exists to mark the test data as having been updated "
                "to match the challenge format.\n\nThis file should not be "
                "removed.")

    # Remove MS COCO directory
    shutil.rmtree("val2017")

    # Rename MS COCO "train2017" to VIPriors "images"
    os.rename("train2017", "images")