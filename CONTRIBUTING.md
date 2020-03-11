# Contributing challenges

Please set up the toolkit for your challenge in the folder named for your challenge. Work on a branch specific to your challenge. Once you completed the toolkit please make a pull request for your branch on GitHub and add another organizer as reviewer.

The scripts we require in each toolkit are:

- A script to transform the full dataset (exactly as downloaded from the public website) to your subset, including training, validation and test subsets.
  - The script should run with minimal provided arguments and require as minimal user effort as possible.
  - Please take care to remove the labels for the test set and obfuscate the filenames of the test samples in your script to make it harder for participants to cheat.
- A submission format definition and/or example submission, to make it easier for challenge partipants to compose their submissions without errors. Submissions should be simply files containing predictions, not code programs.
  - Supplying both a format and an example is of course preferred.
- A baseline method for your challenge. This should include the script to train it, the checkpoint of the model as evaluated and the submission (compatible with evaluation script) as to be uploaded to the CodaLab server.
  - Of course your baseline method also needs to adhere to the rules of the competition: no transfer learning, no fine-tuning.
  - Use your baseline method to test your datasets, submission and evaluation scripts!

We also ask you to implement an evaluation script which takes the example submission and produces scalar metrics for the leaderboards. You can make it compute multiple metrics, but please indicate which metric should be the leading metric. **NOTE that this code AND the testing data (labels) can never be committed/pushed to this repository, as they need to stay private. Please use our private TU Delft repository for this code/data.**
- As this script will be run on the CodaLab servers, it will need to be compatible with their evaluation server environment. The evaluation server runs a Python 3.7 environment. You cannot install packages in it, but is has the following packages installed: `tensorflow, theano==1.0.0, Cython==0.29.13, numpy==1.18.1, scipy==1.3.1, scikit-learn==0.21.3, pandas==0.25.0, pyyaml==5.1, imutils==0.5.3`. *Note that PyTorch is not installed!* If you need PyTorch or some other Python package, let Robert-Jan know, as we can arrange it so that the environment can run custom packages, but it requires some extra work.
- This script will at some point need to adhere to the [API required by the CodaLab evaluation server](https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition). For now it is enough to provide a script that takes a single prediction file and a single groundtruth file as input and produces a `.txt` file with `METRIC_NAME: VALUE` per line (where `VALUE` should be a floating point score). As long as your script matches these constraints we can convert it to the exact format afterwards.

For the structure of your challenge folder you can refer to the `example-challenge` challenge. Adhering to this structure will make the pull request a lot easier. To get started install the dependencies in `example-challenge/requirements.txt`, then fill in the blanks in the provided scripts to implement your toolkit. You can also refer to the `object-detection` challenge for inspiration on how to implement your challenge. Feel free to copy any code from that challenge for implementing your own challenge.