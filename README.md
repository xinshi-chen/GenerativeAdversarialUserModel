# Generative Adversarial User Model

Tensorflow implementation for:

[Generative Adversarial User Model for Reinforcement Learning Based Recommendation System](http://proceedings.mlr.press/v97/chen19f/chen19f.pdf) [1]

(Currently the ant financial dataset is not authorized to released. Experiments on other public dataset are released.)

## Setup

### Install
Clone and install the current package.
```
pip install -e .
```

### Data

The dataset can be obtained via the [shared dropbox folder](https://www.dropbox.com/sh/57gqb1c98gxasr8/AABDPPVnggypWwn2NsLNq7x6a?dl=0).

After downloading the `.txt` files in the shared folder, put then under the 'dropbox' folder, so that the default bash script can automatically find them.

Finally the project has the following folder structure:
```
ganrl
|___ganrl  # source code
|   |___common # common implementations
|   |___experiment_user_model # code for experiments in Sec 6.1 in the paper
|
|___dropbox  # yelp, tb, rsc dataset.
    |___process_data.py
    |___process_data.sh
    |___yelp.txt
    |___tb.txt
    |......
...
```

Process the data before running the experiments:
```
cd dropbox
./process_data.sh
```

## Experiments

By modifying the sh scripts, You can tune the hyperparameters like the architecture of the neural networks, learning rate, etc.

### GA User Model with Shannon Entropy
Navigate to the experiment folder. You can run the sh script directly or set the hyperparameters by yourself.
To try a different split of train, test, validation sets, you can change `-resplit False` to `-resplit True` in the sh file. 
```
cd ganrl/experiment_user_model/
./run_gan_user_model.sh
```
The trained model will be saved in `scratch/` folder.

### GA User Model with L2 Regularization
First, train the user model using Shannon Entropy by running `./run_gan_user_model.sh`. With this saved model as an initilization, you can continue to train the model using other regularizations. For example, L2:
```
cd ganrl/experiment_user_model/
./run_gan_user_model.sh
./run_gan_L2_regularized_yelp.sh
```

## Citation
If you found it useful in your research, please consider citing
```
@inproceedings{chen2019generative,
  title={Generative Adversarial User Model for Reinforcement Learning Based Recommendation System},
  author={Chen, Xinshi and Li, Shuang and Li, Hui and Jiang, Shaohua and Qi, Yuan and Song, Le},
  booktitle={International Conference on Machine Learning},
  pages={1052--1061},
  year={2019}
}
```

## References
[1] Xinshi Chen, Shuang Li, Hui Li, Shaohua Jiang, Yuan Qi, Le Song. "Generative Adversarial User Model for Reinforcement Learning Based Recommendation System." *In International Conference on Machine Learning.* 2019.
