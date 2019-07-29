from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse
import os
import numpy as np

cmd_opt = argparse.ArgumentParser(description='Argparser for GAN user model for RL based recommendation')

cmd_opt.add_argument('-data_folder', type=str, default=None, help='dataset folder')
cmd_opt.add_argument('-dataset', type=str, default=None, help='choose rsc, tb, or yelp')
cmd_opt.add_argument('-save_dir', type=str, default='./scratch', help='save folder')

cmd_opt.add_argument('-resplit', type=eval, default=False)

cmd_opt.add_argument('-num_thread', type=int, default=10, help='number of threadings')
cmd_opt.add_argument('-learning_rate', type=float, default=1e-3, help='learning rate')
cmd_opt.add_argument('-batch_size', type=int, default=128, help='batch size')
cmd_opt.add_argument('-num_itrs', type=int, default=2000, help='num of iterations')

cmd_opt.add_argument('-rnn_hidden_dim', type=int, default=20, help='LSTM hidden sizes')
cmd_opt.add_argument('-pw_dim', type=int, default=4, help='position weight dim')
cmd_opt.add_argument('-pw_band_size', type=int, default=20, help='position weight banded size (i.e. length of history)')

cmd_opt.add_argument('-dims', type=str, default='64-64')
cmd_opt.add_argument('-user_model', type=str, default='LSTM', help='architecture choice: LSTM or PW')


cmd_args = cmd_opt.parse_args()
#
# if cmd_args.save_dir is not None:
#     if not os.path.isdir(cmd_args.save_dir):
#         os.makedirs(cmd_args.save_dir)

print(cmd_args)
