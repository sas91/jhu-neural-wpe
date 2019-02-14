#!/usr/bin/env python

#  2018 JHU CLSP (Aswin Shanmugam Subramanian)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
from scipy.io import wavfile
import pickle
import gzip
import threading


def log_sp(x):
    return np.log(x + 1e-08)


def exp_sp(x):
    return np.exp(x) - 1e-08


def read_audio(filename):
    rate, audio_data = wavfile.read(filename)
    return audio_data


def stack_features(data, context):
    if context == 0:
        return data
    padded = np.r_[np.repeat(data[0][None], context, axis=0), data,
                   np.repeat(data[-1][None], context, axis=0)]
    stacked_features = np.zeros((len(data), (2 * context + 1) *
                                 data.shape[1])).astype(np.float32)
    for i in range(context, len(data) + context):
        sfea = padded[i - context: i + context + 1]
        stacked_features[i - context] = sfea.reshape(-1)
    return stacked_features


def get_batch(file_name):
    f = gzip.open(file_name, 'rb')
    data = pickle.load(f)
    Y = data['s_x_abs_list']
    X = data['s_n_abs_list']
    return X, Y


def wavwrite_scipy(filename, samplerate, enhanced_signal):
    enhanced_signal = enhanced_signal.copy()
    int16_max = np.iinfo(np.int16).max
    int16_min = np.iinfo(np.int16).min
    enhanced_signal *= int16_max
    enhanced_signal = np.clip(enhanced_signal, int16_min, int16_max)
    enhanced_signal = enhanced_signal.astype(np.int16)
    threading.Thread(target=wavfile.write, args=(filename, samplerate,
                                                 enhanced_signal)).start()
