#!/usr/bin/env python

# Copyright 2018 Johns Hopkins University (Author: Aswin Shanmugam Subramanian)
# Apache 2.0
# Based on https://github.com/fgnt/nara_wpe/

import numpy as np
import functools
import operator


def hermite(x):
    return x.swapaxes(-2, -1).conj()


def get_working_shape(shape):
    product = functools.reduce(operator.mul, [1] + list(shape[:-2]))
    return [product] + list(shape[-2:])


def segment_axis(x, length, shift, axis=-1):

    if x.__class__.__module__ == 'cupy.core.core':
        import cupy
        xp = cupy
    else:
        xp = np

    axis = axis % x.ndim
    elements = x.shape[axis]

    if shift <= 0:
        raise ValueError('Can not shift forward by less than 1 element.')

    shape = list(x.shape)
    del shape[axis]
    shape.insert(axis, (elements + shift - length) // shift)
    shape.insert(axis + 1, length)

    strides = list(x.strides)
    strides.insert(axis, shift * strides[axis])

    if xp == np:
        return np.lib.stride_tricks.as_strided(x, strides=strides, shape=shape)
    else:
        x = x.view()
        x._set_shape_and_strides(strides=strides, shape=shape)
        return x


def build_y_tilde(Y, taps, delay):
    S = Y.shape[:-2]
    D = Y.shape[-2]
    T = Y.shape[-1]

    def pad(x, axis=-1, pad_width=taps + delay - 1):
        npad = np.zeros([x.ndim, 2], dtype=np.int)
        npad[axis, 0] = pad_width
        x = np.pad(x,
                   pad_width=npad,
                   mode='constant',
                   constant_values=0)
        return x

    Y_ = pad(Y)
    Y_ = np.moveaxis(Y_, -1, -2)
    Y_ = np.flip(Y_, axis=-1)
    Y_ = np.ascontiguousarray(Y_)
    Y_ = np.flip(Y_, axis=-1)
    Y_ = segment_axis(Y_, taps, 1, axis=-2)
    Y_ = np.flip(Y_, axis=-2)
    if delay > 0:
        Y_ = Y_[..., :-delay, :, :]
    Y_ = np.reshape(Y_, list(S) + [T, taps * D])
    Y_ = np.moveaxis(Y_, -2, -1)
    return Y_


def _stable_solve(A, B):
    assert A.shape[:-2] == B.shape[:-2], (A.shape, B.shape)
    assert A.shape[-1] == B.shape[-2], (A.shape, B.shape)
    try:
        return np.linalg.solve(A, B)
    except np.linalg.linalg.LinAlgError:
        shape_A, shape_B = A.shape, B.shape
        assert shape_A[:-2] == shape_A[:-2]
        working_shape_A = get_working_shape(shape_A)
        working_shape_B = get_working_shape(shape_B)
        A = A.reshape(working_shape_A)
        B = B.reshape(working_shape_B)

        C = np.zeros_like(B)
        for i in range(working_shape_A[0]):
            try:
                C[i] = np.linalg.solve(A[i], B[i])
            except np.linalg.linalg.LinAlgError:
                C[i] = np.linalg.lstsq(A[i], B[i])[0]
        return C.reshape(*shape_B)


def wpe_filter(Y, X, taps=10, delay=3):
    Y_tilde = build_y_tilde(Y, taps, delay)
    power = np.mean(X, axis=-2)
    eps = 1e-10 * np.max(power)
    inverse_power = 1 / np.maximum(power, eps)
    Y_tilde_inverse_power = Y_tilde * inverse_power[..., None, :]
    R = np.matmul(Y_tilde_inverse_power, hermite(Y_tilde))
    P = np.matmul(Y_tilde_inverse_power, hermite(Y))
    G = _stable_solve(R, P)
    X = Y - np.matmul(hermite(G), Y_tilde)
    return X
