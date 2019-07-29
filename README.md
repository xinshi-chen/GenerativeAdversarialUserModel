# Generative Adversarial User Model

Tensorflow implementation for:

[Generative Adversarial User Model for Reinforcement Learning Based Recommendation System](https://arxiv.org/abs/1812.10613) [1]

(Currently the ant financial dataset is not authorized to released. Experiments on other public dataset are released.)

# Setup

## Install
Clone and install the current package.
```
pip install -e .
```

## data

The dataset can be obtained via the [shared dropbox folder](https://www.dropbox.com/sh/57gqb1c98gxasr8/AABDPPVnggypWwn2NsLNq7x6a?dl=0)

After downloading the shared folder, put it under the root of the project (or create a symbolic link) and rename it as 'dropbox', so that the default bash script can automatically find them.

Finally the project has the following folder structure:
```
ganrl
|___ganrl  # source code
|   |___common # common implementations
|   |___experiment_user_model # code for experiments in Sec 6.1 in the paper
|
|___dropbox  # processed yelp, tb, rsc dataset.
    |___yelp.txt
    |___tb.txt
    |......
...
```

# References
[1] Xinshi Chen, Shuang Li, Hui Li, Shaohua Jiang, Yuan Qi, Le Song. "Generative Adversarial User Model for Reinforcement Learning Based Recommendation System." *In International Conference on Machine Learning.* 2019.
