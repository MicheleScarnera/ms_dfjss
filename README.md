# ms_dfjss
An implementation of the DFJSS (Dynamic Flexible Job Shop Scheduling) problem [(cit)](https://ietresearch.onlinelibrary.wiley.com/doi/abs/10.1049/iet-cim.2018.0009), used for my Master's Thesis.

For additional context, you can find its PDF among these files.

The main object of study in the problem are [priority functions](https://ieeexplore.ieee.org/abstract/document/9234005?casa_token=NulXPxgNdZcAAAAA:LCJDu4JDkLKYJtVLbc-SpZTSV9rbhrjGMXopQ0aVDPSzfvRyTrW90PPG7rNgWzZbNgx0HdVFUlSgEw), which are a heuristic that sort all the possible actions by their "priority", defined by the function.
A priority function is deemed better than another if the resulting schedule results in a lower loss function value.

This code also implements random job arrival and machine breakdown.

The optimization routines used for finding optimal priority functions are [genetic algorithms](http://www.gp-field-guide.org.uk), and an auto-encoder/feed-forward network system, which is explained in the thesis.

A brief explanation of the `dfjss_run_` scripts:
- Run `dfjss_run_warehouse` for a verbose DFJSS simulation;
- Run `dfjss_run_genetic` for the genetic algorithm;
- Run `dfjss_run_autoencoder` for the training of the priority functions' autoencoder;
- Run `dfjss_run_reward_model` for the training of the feed-forward reward model;
  - Run `dfjss_run_reward_model_dataset` to generate its data. Note that the compressed archive `reward_model_dataset.7z` contains the dataset file that was used for the thesis. You may use it if you extract it in the root folder.
- Run `dfjss_run_reward_model_optimizer` to optimize the reward from the reward model;
- Run `dfjss_run_all_plots` to generate the plots used in the thesis.

This was written in `Python 3.11`, using the following packages:
```
numpy
scipy
torch
matplotlib
```