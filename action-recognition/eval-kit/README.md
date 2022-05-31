# VIPriors Action Recognition Challenge - Evaluation

The evaluation of the challenge is hosted in CodaLab. Participating in the challenge requires a CodaLab account. Please, find the evaluation server [here](link-here).

## Metric

For this task we will evaluate the average classification accuracy over all classes on the test set. The accuracy for one class is calculated using the next equation:
$$
\mathrm{Acc} = \frac{P}{N},
$$
where P corresponds to the number of correct predictions for the class being evaluated and N to the total number of samples of the class. The average accuracy is the average of accuracies over all classes.

The winner of the challenge will be determined by the highest Top-1 average accuracy. However, as extra information of the performance, we will also compute the Top-3 and Top-5 average accuracy.

## Code

We provide here the code that runs in the evaluation server. (`eval_kinetics400ViPriors.py`).  To use it just do the following:

`python eval_kinetics400ViPriors.py -pred <filepath> -gt <filepath> -outdir <dirpath> -topk <list>`

If you need more help about the arguments, just type the following command and some help will be printed in the prompt.

`python eval_kinetics400ViPriors.py -h`

## Submission file format

Please, be aware of submitting your results in a .txt file with the appropriate format:

```
videoid cls1 cls2 cls3 cls4 cls5 ...
videoid cls1 cls2 cls3 cls4 cls5 ...
...
```

Additionally, the following restrictions are implemented in the evaluation server:

- All classes are integer numbers in the range [1, 400]. This means only integer numbers are read from the result file. Any number out of the range will be considered as a classification error.
- If no predicted class is found in the file, the evaluation code will consider it as a classification error.
- Put as many class predictions as you wish in the result file. The evaluation server will take into account only the first 5 predictions.

You can run the baseline to get an example of this file. Please, take it as a guide.

## Additional information

An updated leaderboard with each participant's score in the test set will be shown during the time the challenge is opened.

We encourage the people to proceed this way (remember to follow challenge's restrictions):

1. Train your model from scratch in the train set.
2. Validate your model with the validation set.
3. Iterate over 1 and 2 to find your best model.
4. Run your model on the test set.
5. Submit your results to the server and see your score.

