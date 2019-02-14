#!/usr/bin/env python

#  2018 JHU CLSP (Aswin Shanmugam Subramanian)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import os
import numpy as np
import chainer
from chainer import Variable
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from tqdm import tqdm
from utils import stack_features, get_batch
from dnn_model import LSTM_dereverb

parser = argparse.ArgumentParser(description='LSTM_dereverb training')
parser.add_argument('--data_dir', default='data',
                    help='Directory used for the training data')
parser.add_argument('--model_dir', default='model',
                    help='Directory used to store the model file')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--num_epochs', default=30, type=int,
                    help='Number of epochs to train')
parser.add_argument('--patience', default=5, type=int,
                    help='Max. number of epochs to wait for better CV loss')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
args = parser.parse_args()

log = logging.getLogger('dnn_wpe')
log.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(args.data_dir, 'dnn_wpe.log'))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
log.addHandler(fh)
log.addHandler(ch)

model = LSTM_dereverb()
model_save_dir = args.model_dir
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy

file_lists = dict()
for data_type in ['tr', 'dt']:
    file_name = os.path.join(args.data_dir, 'file_list_{}'.format(data_type))
    with open(file_name, 'r') as in_file:
        file_list = in_file.read().split('\n')
    del file_list[-1]
    file_lists[data_type] = file_list

optimizer = optimizers.Adam()
optimizer.setup(model)

epoch = 0
exhausted = False
best_epoch = 0
best_cv_loss = np.inf
max_epochs = args.num_epochs

if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_hdf5(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_hdf5(args.resume, optimizer)

while (epoch < max_epochs and not exhausted):
    log.info('Epoch {}'.format(epoch))
    perm = np.random.permutation(len(file_lists['tr']))
    sum_loss_tr = 0
    with chainer.using_config("train", True):
        for i in tqdm(range(0, len(file_lists['tr'])),
                      desc='Epoch {} - Training'.format(epoch)):
            x_var = []
            y_var = []
            X, Y = get_batch(file_lists['tr'][perm[i]])
            for x_var_temp in X:
                x_var_temp = stack_features(x_var_temp, 5)
                x_var_temp = Variable(x_var_temp)
                if args.gpu >= 0:
                    x_var_temp.to_gpu(args.gpu)
                x_var.append(x_var_temp)
            y_var = Variable(np.vstack(Y))
            if args.gpu >= 0:
                y_var.to_gpu(args.gpu)
            loss = model(x_var, y_var)
            optimizer.target.cleargrads()
            loss.backward()
            optimizer.update()
            sum_loss_tr += float(loss.data)

    sum_loss_cv = 0
    with chainer.using_config("train", False):
        with chainer.no_backprop_mode():
            for i in tqdm(range(0, len(file_lists['dt'])),
                          desc='Epoch {} - CV'.format(epoch)):
                x_var = []
                X, Y = get_batch(file_lists['dt'][i])
                for x_var_temp in X:
                    x_var_temp = stack_features(x_var_temp, 5)
                    x_var_temp = Variable(x_var_temp)
                    if args.gpu >= 0:
                        x_var_temp.to_gpu(args.gpu)
                    x_var.append(x_var_temp)
                y_var = Variable(np.vstack(Y))
                if args.gpu >= 0:
                    y_var.to_gpu(args.gpu)
                loss = model(x_var, y_var)
                sum_loss_cv += float(loss.data)

    loss_tr = (sum_loss_tr) / len(file_lists['tr'])
    loss_cv = (sum_loss_cv) / len(file_lists['dt'])

    log.info('Mean loss at epoch {} during training: {:.5f}, cross-validation: {:.5f}'.
             format(epoch, loss_tr, loss_cv))

    if loss_cv < best_cv_loss:
        best_epoch = epoch
        best_cv_loss = loss_cv
        model_file = os.path.join(model_save_dir, 'best.nnet')
        log.info('Improved model at epoch {}'.format(epoch))
        serializers.save_hdf5(model_file, model)
        serializers.save_hdf5(os.path.join(model_save_dir,
                                           'mlp.tr'), optimizer)

    if epoch - best_epoch == args.patience:
        exhausted = True
        log.info('Patience exhausted. Stopping training')

    epoch += 1

log.info('Finished!')
