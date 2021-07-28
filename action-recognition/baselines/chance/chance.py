"""
This script generates a random chance baseline on the validation set.

Usage:
    python chance.py
"""
import random


if __name__ == "__main__":
    # Validation file
    ann_dir = "../../data/annotations/"
    val_file = f"{ann_dir}kinetics400ViPriors-val.csv"

    # Number of classes and number of opportunities
    n_cls = 400
    top_k = 5

    # Chance result file
    chance_file = "./chance_baseline_val.txt"

    # Open validation file and read lines
    with open(val_file, "r") as train_f:
        lines = train_f.readlines()
    lines.pop(0)

    # Generate k random classes for each clip
    clip_lst = list()
    cls_lst = list()
    random.seed(1992)
    for line in lines:
        chance = list()
        for k in range(0, top_k, 1):
            chance.append(random.randint(1, n_cls + 1))
        clip_lst.append(line.split(",")[1])
        cls_lst.append(chance)
	
	# Randomising clips
    random.seed(1992)
    random.shuffle(clip_lst)
	
    # Generate files with results
    with open(chance_file, 'w') as chance_f:
        for i, clip in enumerate(clip_lst):
            line = f'{clip} {cls_lst[i][0]} {cls_lst[i][1]} {cls_lst[i][2]} ' \
                   f'{cls_lst[i][3]} {cls_lst[i][4]}\n'
            chance_f.write(line)

    # Finished
    print("Random chanche baseline on Validation set generated.")

