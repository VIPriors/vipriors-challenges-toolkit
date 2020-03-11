# VIPriors Action Recognition Challenge - Baselines

We provide different baselines as a starting point for participants to compare their codes to. The results of all the baselines are for the validation set of the challenge dataset. We also provide tooling to generate the results.

All baselines are generated following the restrictions and conditions of the challenge. Please, make sure to also follow the rules. 

## Baselines

The baselines we offer are:

- **Random chance baseline.** It's a baseline that simply assigns random labels to clips. The script to generate the results of this baseline can be found in the `chance/` folder, inside this directory.
- ~~**C3D baseline.** We implemented in PyTorch the C3D model designed by [Du Tran et al.](https://arxiv.org/pdf/1412.0767v4.pdf) All related files to this implementation can be found in the `c3d/` folder, inside this directory.~~ *(C3D baseline will be released soon)*

## Performance

We show, in the table below, the performance of the baselines on the validation set

| Baseline | Top-1 Accuracy (%) | Top-3 Accuracy (%) | Top-5 Accuracy (%) |
| :------: | :----------------: | :----------------: | :----------------: |
|  chance  |        1.1         |        2.83        |        5.00        |
|   C3D    |         X          |         X          |         X          |
