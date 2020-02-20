"""
The evaluation script goes here. It should store predictions in the file
`scores.txt`. The metric saving code is given already.

Input:
    predictions (str): filepath;
    groundtruths (str): filepath;

Usage:
    evaluate.py <groundtruths> <predictions>
"""
import docopt

OUTPUT_FILE = 'scores.txt'


def evaluate(args):
    with open(args['<groundtruths>'], 'r') as f:
        # TODO: read groundtruths contents
        pass

    with open(args['<predictions>'], 'r') as f:
        # TODO: read predictions contents
        pass

    # TODO: evaluation code
    pass

    # Write metrics to file
    # NOTE(rjbruin): make sure to store metrics as a list of tuples
    # (name (str), value (float))
    metrics = []
    with open(OUTPUT_FILE, 'w') as f:
        for name, val in metrics:
            f.write(f"{name}: {val:.8f}\n")

if __name__ == '__main__':
    args = docopt.docopt(__doc__, version='1.0.0')
    evaluate(args)