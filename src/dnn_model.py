#!/usr/bin/env python

#  2018 JHU CLSP (Aswin Shanmugam Subramanian)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import chainer.functions as F
import chainer.links as L
import chainer


class LSTM_dereverb(chainer.Chain):

    def __init__(self, nfft=257, context=5):
        super(LSTM_dereverb, self).__init__()
        with self.init_scope():
            self.lstm = L.NStepLSTM(1, (nfft*2*context)+nfft, 500, 0)
            self.ll1 = L.Linear(500, 2048)
            self.ll2 = L.Linear(2048, nfft)

    # This function is for forward propagation
    def __call__(self, xs, ts):
        hy, cy, ys = self.lstm(None, None, xs)
        ys = F.relu(self.ll1(F.vstack(ys)))
        del hy, cy
        ys = self.ll2(ys)
        loss = F.mean_squared_error(ys, ts)
        return loss

    # This is used during testing
    def predict(self, xs):
        hy, cy, ys = self.lstm(None, None, xs)
        ys = F.relu(self.ll1(F.vstack(ys)))
        del hy, cy
        ys = self.ll2(ys)
        return ys
