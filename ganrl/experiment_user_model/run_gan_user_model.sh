#!/usr/bin/env bash

dropbox=../../dropbox
data_dir=$dropbox/

user_model=LSTM
dataset=yelp

subdir=model-${user_model}-dataset-${dataset}
save_dir=scratch/$subdir

python main_gan_user_model.py \
        -user_model $user_model \
        -dataset $dataset \
        -save_dir $save_dir \
        -data_folder $data_dir \
        -num_thread 10 \
        -rnn_hidden_dim 20 \
        -dims 64-64 \
        -learning_rate 0.0005 \
        -batch_size 50 \
        -num_itrs 2000 \
        -resplit False \
        -pw_dim 4 \
    $@
