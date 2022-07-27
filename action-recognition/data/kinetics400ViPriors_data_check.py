from pathlib import Path
import cv2


def check_video(filename, my_corrupted_videos_file):

    vid = cv2.VideoCapture(str(filename))
    if not vid.isOpened():
        with open(my_corrupted_videos_file, "a") as my_corrupted_videos_f:
            my_corrupted_videos_f.write(f"{Path(*filename.parts[-3:])}\n")


def compare_corrupted_videos(my_corrupted_videos_file, vipriors_corrupted_videos_file):

    with open(my_corrupted_videos_file) as f:
        my_corrupted_videos = f.read().splitlines()
    my_names = []
    for vid in my_corrupted_videos:
        my_names.append(Path(vid).name)

    with open(vipriors_corrupted_videos_file) as f:
        vipriors_corrupted_videos = f.read().splitlines()
    vipriors_names = []
    for vid in vipriors_corrupted_videos:
        vipriors_names.append(Path(vid).name)
    n_vipriors_names = len(vipriors_names)

    n_same_corrupted_names = len(set(vipriors_names).intersection(set(my_names)))
    if n_same_corrupted_names == n_vipriors_names:
        print("Your version of Kinetics400ViPriors dataset is correct.")
    else:
        print("Check your download.")


def check_dataset():

    subset_files = {"train": 40000,
                    "val": 10000,
                    "test": 20000
                    }

    current_path = Path().absolute()
    my_corrupted_videos_file = current_path / "my_corrupted_videos.txt"
    if my_corrupted_videos_file.exists():
        my_corrupted_videos_file.unlink()

    for subset, n_files in subset_files.items():
        if not (current_path / subset).exists():
            print(f"{subset} subset dir not found.")
            return

        vids_in_dir = list((current_path / subset).glob("*.mp4"))
        if len(vids_in_dir) != n_files:
            print(f"The number of videos for {subset} subset is not correct.")

        for vid in vids_in_dir:
            print(f"Checking {vid}.")
            check_video(vid, my_corrupted_videos_file)

    vipriors_corrupted_videos_file = Path("./annotations/kinetics400ViPriors_corrupted_videos.txt")
    compare_corrupted_videos(my_corrupted_videos_file, vipriors_corrupted_videos_file)

    print("\nDONE!")


if __name__ == '__main__':
    check_dataset()