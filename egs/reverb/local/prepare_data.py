#!/usr/bin/env python

#  2018 JHU CLSP (Aswin Shanmugam Subramanian)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import os
import numpy as np
from tqdm import tqdm
from scipy import signal
import gzip
import pickle
from utils import read_audio, log_sp

parser = argparse.ArgumentParser(description='Prepare training data')
parser.add_argument('--list_tr', default='data/list_tr',
                    help='training file list')
parser.add_argument('--list_dt', default='data/list_dt',
                    help='dev file list')
parser.add_argument('--data_dir', default='data',
                    help='Directory used for the training data '
                    'and to store the model file.')
parser.add_argument('--thread_id', default='1',
                    help='integer thread number')
parser.add_argument('--batch_size', default='5', type=int,
                    help='integer thread number')
args = parser.parse_args()

for data_type in ['tr', 'dt']:
    train_file_list = list()
    if not os.path.exists(os.path.join(args.data_dir, data_type)):
        try:
            os.makedirs(os.path.join(args.data_dir, data_type))
        except OSError:
            if not os.path.isdir(os.path.join(args.data_dir, data_type)):
                raise
    if data_type in ['tr']:
        file_name = args.list_tr
    if data_type in ['dt']:
        file_name = args.list_dt
    with open(file_name, 'r') as in_file:
        file_list = in_file.read().split('\n')
    del file_list[-1]
    perm = np.random.permutation(len(file_list))
    bno = 0
    for i in tqdm(range(0, len(file_list), args.batch_size),
                  desc='Generating data for {}'.format(data_type)):
        bno = bno + 1
        s_n_abs_list = []
        s_x_abs_list = []
        for bid in range(0, args.batch_size):
            if i+bid < len(file_list):
                f_template = file_list[perm[i+bid]]
                for ch in range(1, 9):
                    f = f_template + '_ch{}.wav'.format(ch)
                    f_no_ltr = f_template + '_ch{}.NLR.wav'.format(ch)
                    ltr_audio = read_audio(f)
                    no_ltr_audio = read_audio(f_no_ltr)
                    fx, tx, s_x = signal.stft(no_ltr_audio, fs=16000,
                                              nperseg=512,
                                              noverlap=512-128, nfft=512)
                    fn, tn, s_n = signal.stft(ltr_audio, fs=16000,
                                              nperseg=512,
                                              noverlap=512-128, nfft=512)
                    s_x = np.transpose(s_x)
                    s_n = np.transpose(s_n)
                    s_x_abs = 20 * log_sp(np.abs(s_x))
                    s_n_abs = 20 * log_sp(np.abs(s_n))
                    s_x_abs_list.append(s_x_abs.astype(np.float32))
                    s_n_abs_list.append(s_n_abs.astype(np.float32))
        train_dict = {
            's_x_abs_list': s_x_abs_list,
            's_n_abs_list': s_n_abs_list
        }
        train_name = os.path.join(args.data_dir, data_type,
                                  '{}_batch_{}.{}'.format(args.thread_id,
                                                          bno, 'pklz'))
        fid = gzip.open(train_name, 'wb')
        pickle.dump(train_dict, fid)
        train_file_list.append(train_name)
    with open(os.path.join(args.data_dir, 'file_list_{}_{}'.
                           format(data_type, args.thread_id)), 'w') as fid:
        for f in train_file_list:
            fid.write(f + '\n')
