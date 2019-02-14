#!/usr/bin/env python

#  2018 JHU CLSP (Aswin Shanmugam Subramanian)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import os
import numpy as np
from chainer import Variable
from chainer import cuda
from chainer import serializers
from tqdm import tqdm
from scipy import signal
import chainer
from utils import log_sp, exp_sp, read_audio, stack_features, wavwrite_scipy
from dnn_model import LSTM_dereverb
from wpe import wpe_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='DNN WPE Dereverberation')
parser.add_argument('input_file_name',
                    help='input_file_name')
parser.add_argument('output_file_name',
                    help='output_file_name')
parser.add_argument('--model', '-m', default='model/best.nnet', type=str,
                    help='Trained model file')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--single', '-s', default=0, type=int,
                    help='0 for multi-channel and 1 for single channel')
parser.add_argument('--is_real', '-r', default=0, type=int,
                    help='0 for simu and 1 for real')
args = parser.parse_args()

model = LSTM_dereverb()

serializers.load_hdf5(args.model, model)
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy

with open(args.input_file_name, 'r') as in_file:
    flist = in_file.read().split('\n')
del flist[-1]
with open(args.output_file_name, 'r') as in_file:
    flist_out = in_file.read().split('\n')

for i in tqdm(range(0, len(flist))):
    cur_line = flist[i]
    s_x_abs_list = []
    s_x_list = []
    if args.single == 0:
        dir_path = os.path.dirname(cur_line)
        f_name = os.path.basename(cur_line)
        for ch in range(1, 9):
            if args.is_real == 1:
                dir_path = os.path.dirname(cur_line)
                f_name = os.path.basename(cur_line)
                part1 = f_name.split('-')[0]
                part2 = f_name.split('-')[1]
                part3 = f_name.split('-')[2].split('_')[1]
                f_name_new = part1 + '-' + part2 + '-' + '{}'.format(ch) + '_' + part3
                f = os.path.join(dir_path, f_name_new)
            elif args.is_real == 0:
                f_name_no_ch = f_name.split('_')[0]
                f_with_ch = f_name_no_ch + '_ch{}.wav'.format(ch)
                f = os.path.join(dir_path, f_with_ch)
            audio_data = read_audio(f)
            fx, tx, s_x = signal.stft(audio_data, fs=16000, nperseg=512,
                                      noverlap=512-128, nfft=512)
            s_x = np.transpose(s_x)
            if ch == 1:
                T, F = s_x.shape
                Y = np.zeros((8, T, F), dtype=np.complex64)
            Y[ch-1, :, :] = s_x
            s_x_abs = 20 * log_sp(np.abs(s_x))
            s_x_abs = stack_features(s_x_abs.astype(np.float32), 5)
            s_x_abs = Variable(s_x_abs)
            if args.gpu >= 0:
                s_x_abs.to_gpu(args.gpu)
            s_x_abs_list.append(s_x_abs)
    elif args.single == 1:
        audio_data = read_audio(cur_line)
        fx, tx, s_x = signal.stft(audio_data, fs=16000, nperseg=512,
                                  noverlap=512-128, nfft=512)
        s_x = np.transpose(s_x)
        s_x_abs = 20 * log_sp(np.abs(s_x))
        s_x_abs = stack_features(s_x_abs.astype(np.float32), 5)
        s_x_abs = Variable(s_x_abs)
        if args.gpu >= 0:
            s_x_abs.to_gpu(args.gpu)
        s_x_abs_list.append(s_x_abs)
    with chainer.no_backprop_mode():
        X = model.predict(s_x_abs_list)
        if args.gpu >= 0:
            X.to_cpu()

    if args.single == 0:
        xs = X.data
        xs = np.reshape(xs, (8, xs.shape[0] // 8, xs.shape[1]))
        taps = 10
        xs_power = np.square(exp_sp(xs/20))
        Y_hat = np.copy(Y)
        D, T, F = xs.shape
        for f in range(0, F):
            Y_hat[:, :, f] = wpe_filter(Y[:, :, f], xs_power[:, :, f],
                                        taps=taps, delay=3)

    if i == 0 and args.single == 0:
        v_min = np.min(np.array([20 * log_sp(np.abs(s_x)), xs[7, :, :],
                       20 * log_sp(np.abs(Y_hat[7, :, :]))]))
        v_max = np.max(np.array([20 * log_sp(np.abs(s_x)), xs[7, :, :],
                       20 * log_sp(np.abs(Y_hat[7, :, :]))]))
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.pcolormesh(tx, fx, np.transpose(20*log_sp(np.abs(s_x))),
                       cmap='jet', vmin=v_min, vmax=v_max)
        plt.title('Noisy Eval Example')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        clb = plt.colorbar()
        clb.set_label('Decibels (dB)')
        plt.subplot(3, 1, 2)
        plt.pcolormesh(tx, fx, np.transpose(xs[7, :, :]), cmap='jet',
                       vmin=v_min, vmax=v_max)
        plt.title('DNN Output')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        clb = plt.colorbar()
        plt.tight_layout()
        clb.set_label('Decibels (dB)')
        plt.subplot(3, 1, 3)
        plt.pcolormesh(tx, fx, np.transpose(20*log_sp(np.abs(Y_hat[7, :, :]))),
                       cmap='jet', vmin=v_min, vmax=v_max)
        plt.title('DNN-WPE Output')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        clb = plt.colorbar()
        plt.tight_layout()
        clb.set_label('Decibels (dB)')
        plt.savefig('data/plot_example.png')

    if args.single == 0:
        for ch in range(1, 9):
            tx, enhanced_signal = signal.istft(np.transpose(Y_hat[ch-1, :, :]),
                                               fs=16000, nperseg=512,
                                               noverlap=512-128, nfft=512)
            enhanced_signal /= np.max(np.abs(enhanced_signal))
            dir_path = os.path.dirname(flist_out[i])
            if args.is_real == 0:
                f_with_ch = f_name_no_ch + '_ch{}.wav'.format(ch)
            elif args.is_real == 1:
                f_with_ch = part1 + '-' + part2 + '-' + '{}'.format(ch) + '_' + part3
            f_out = os.path.join(dir_path, f_with_ch)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            wavwrite_scipy(f_out, 16000, enhanced_signal)
    elif args.single == 1:
        Y_hat = exp_sp(np.transpose(X.data)/20) * np.exp(1j*np.angle(np.transpose(s_x)))
        tx, enhanced_signal = signal.istft(Y_hat, fs=16000, nperseg=512,
                                           noverlap=512-128, nfft=512)
        enhanced_signal /= np.max(np.abs(enhanced_signal))
        dir_path = os.path.dirname(flist_out[i])
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        wavwrite_scipy(flist_out[i], 16000, enhanced_signal)

print('Finished')
