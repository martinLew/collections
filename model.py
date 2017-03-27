"""
Definition of a mlp net for IF price modeling
"""

from __future__ import division
from __future__ import print_function

import logging
import math
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
# from chainer.initializers import normal

logger = logging.getLogger(__name__)


class MultiOut(chainer.ChainList):
    """Model output layer, which may give out more than 1 output"""
    def __init__(self, in_size, n_out=1, out_size=3):
        layers = [L.Linear(in_size, out_size) for i in range(n_out)]
        super(MultiOut, self).__init__(*layers)

    def __call__(self, x):
        return [l(x) for l in self]


class MLP(chainer.Chain):

    def __init__(self, in_size, n_units, n_out=1, out_size=3, train=True,
                 batch_norm=False):
        super(MLP, self).__init__(
            l1=L.Linear(in_size, n_units),
            bn1=L.BatchNormalization(n_units),
            l2=L.Linear(n_units, n_units),
            bn2=L.BatchNormalization(n_units),
            out=MultiOut(n_units, n_out, out_size),
        )

        self.train = train
        self.target_num = n_out
        self.batch_norm = batch_norm

    def __call__(self, x):
        if self.batch_norm:
            h1 = F.relu(self.bn1(self.l1(x), test=not self.train))
            h2 = F.relu(self.bn2(self.l2(h1), test=not self.train))
        else:
            h1 = F.relu(self.l1(x))
            h2 = F.relu(self.l2(h1))

        y = self.out(F.dropout(h2, train=self.train))
        return y


class LSTM(chainer.Chain):

    def __init__(self, in_size, n_units1, n_units2, n_out=1, out_size=3,
                 train=True, batch_norm=False, layer_norm=False):
        super(LSTM, self).__init__(
            l1=L.Linear(in_size, n_units1),
            bn1=L.BatchNormalization(n_units1),
            l2=L.LSTM(n_units1, n_units2),
            ln1=L.LayerNormalization(n_units2),
            out=MultiOut(n_units2, n_out, out_size),
        )

        self.train = train
        self.target_num = n_out
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

    def reset_state(self):
        self.l2.reset_state()

    def __call__(self, x):
        if self.batch_norm:
            h1 = F.relu(self.bn1(self.l1(x), test=not self.train))
        else:
            h1 = F.relu(self.l1(x))

        if self.layer_norm:
            h2 = self.ln1(self.l2(h1))
        else:
            h2 = self.l2(h1)

        y = self.out(F.dropout(h2, train=self.train))
        return y


class LSTMPlus(chainer.Chain):

    def __init__(self, in_size, n_units1, n_units2, n_out=1, out_size=3,
                 train=True, batch_norm=False, layer_norm=False):
        super(LSTMPlus, self).__init__(
            l1=L.Linear(in_size, n_units1),
            bn1=L.BatchNormalization(n_units1),
            l2=L.LSTM(n_units1, n_units2),
            ln1=L.LayerNormalization(n_units2),
            out=MultiOut(n_units2+15, n_out, out_size),
        )

        self.train = train
        self.target_num = n_out
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

    def reset_state(self):
        self.l2.reset_state()

    def __call__(self, x, xplus):
        if self.batch_norm:
            h1 = F.relu(self.bn1(self.l1(x), test=not self.train))
        else:
            h1 = F.relu(self.l1(x))

        if self.layer_norm:
            h1 = self.ln1(self.l2(h1))
        else:
            h1 = self.l2(h1)

        h1 = F.concat([h1, xplus], axis=1)
        y = self.out(F.dropout(h1, train=self.train))
        return y


class BottleNeckA(chainer.Chain):
    def __init__(self, in_size, out_size, stride=2):
        w = math.sqrt(2)
        # w = normal.HeNormal(scale=1.0)
        super(BottleNeckA, self).__init__(
            conv1=L.Convolution2D(in_size, out_size, (1, 3),
                                  stride, (0, 1), w, nobias=True),
            bn1=L.BatchNormalization(out_size),
            conv2=L.Convolution2D(out_size, out_size, (1, 3),
                                  1, (0, 1), w, nobias=True),
            bn2=L.BatchNormalization(out_size),

            conv3=L.Convolution2D(in_size, out_size, 1,
                                  stride, 0, w, nobias=True),
            bn3=L.BatchNormalization(out_size),
        )

    def __call__(self, x, train):
        h1 = self.conv1(x)
        # h1 = self.bn1(h1, test=not train)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, train=train)
        h1 = self.conv2(h1)
        # h1 = self.bn2(h1, test=not train)
        h2 = self.conv3(x)
        # h2 = self.bn3(h2, test=not train)

        return F.relu(h1 + h2)


class BottleNeckB(chainer.Chain):
    def __init__(self, in_size, out_size):
        w = math.sqrt(2)
        # w = normal.HeNormal(scale=1.0)
        super(BottleNeckB, self).__init__(
            conv1=L.Convolution2D(in_size, out_size, (1, 3), 1, (0, 1),
                                  w, nobias=True),
            bn1=L.BatchNormalization(out_size),
            conv2=L.Convolution2D(out_size, out_size, (1, 3), 1, (0, 1),
                                  w, nobias=True),
            bn2=L.BatchNormalization(out_size),
        )

    def __call__(self, x, train):
        h = self.conv1(x)
        # h = self.bn1(h, test=not train)
        h = F.relu(h)
        h = F.dropout(h, train=train)
        h = self.conv2(h)
        # h = self.bn1(h, test=not train)

        return F.relu(h + x)


class Block(chainer.Chain):
    def __init__(self, layer, in_size, out_size, stride=2):
        super(Block, self).__init__()
        links = [('a', BottleNeckA(in_size, out_size, stride))]
        for i in range(layer-1):
            links += [('b{}'.format(i+1), BottleNeckB(out_size, out_size))]

        for l in links:
            self.add_link(*l)
        self.forward = links

    def __call__(self, x, train):
        for name, func in self.forward:
            x = func(x, train)
        return x


class ResNet(chainer.Chain):

    def __init__(self, window_len, feat_num):
        w = math.sqrt(2)
        self.window_len = window_len
        self.feat_num = feat_num
        self.train = True
        self.target_num = 1
        # w = normal.HeNormal(scale=1.0)
        super(ResNet, self).__init__(
            conv1=L.Convolution2D(1, 64, (feat_num, 7),
                                  1, (0, 3), w, nobias=True),
            bn1=L.BatchNormalization(64),
            res2=Block(3, 64, 128, 1),
            res3=Block(6, 128, 256, 2),
            res4=Block(3, 256, 512, 2),
            fc=MultiOut(512, 1, 3))

    def precall(self, xs):
        xs = xs.reshape(xs.shape[0], self.window_len, -1)
        # NOTE: change it in net
        xs = xs.swapaxes(1, 2)
        xs = np.expand_dims(xs, 1)
        return xs

    def __call__(self, x):
        x = self.precall(x)
        h = self.conv1(x)
        h = self.bn1(h, test=not self.train)
        h = F.max_pooling_2d(F.relu(h), (1, 3), stride=2)
        h = self.res2(h, self.train)
        h = self.res3(h, self.train)
        h = self.res4(h, self.train)
        h = F.average_pooling_2d(h, (1, 3), stride=1)
        h = F.dropout(h, train=self.train)
        h = self.fc(h)

        return h


class WBottleNeckA(chainer.Chain):
    def __init__(self, in_size, out_size, stride=2):
        w = math.sqrt(2)
        super(WBottleNeckA, self).__init__(
            bn1=L.BatchNormalization(in_size),
            conv1=L.Convolution2D(in_size, out_size, (1, 3),
                                  stride, (0, 1), w, nobias=True),

            bn2=L.BatchNormalization(out_size),
            conv2=L.Convolution2D(out_size, out_size, (1, 3),
                                  1, (0, 1), w, nobias=True),

            conv3=L.Convolution2D(in_size, out_size, 1,
                                  stride, 0, w, nobias=True),
        )

    def __call__(self, x, train):
        h1 = self.conv1(F.dropout(F.relu(self.bn1(x,
                        test=not train)), train=train))
        h1 = self.conv2(F.relu(self.bn2(h1, test=not train)))
        h2 = self.conv3(x)
        return F.relu(h1 + h2)


class WBottleNeckB(chainer.Chain):
    def __init__(self, in_size, out_size):
        w = math.sqrt(2)
        super(WBottleNeckB, self).__init__(
            bn1=L.BatchNormalization(in_size),
            conv1=L.Convolution2D(in_size, out_size, (1, 3), 1, (0, 1),
                                  w, nobias=True),
            bn2=L.BatchNormalization(out_size),
            conv2=L.Convolution2D(out_size, out_size, (1, 3), 1, (0, 1),
                                  w, nobias=True),
        )

    def __call__(self, x, train):
        h = self.conv1(F.dropout(F.relu(self.bn1(x,
                       test=not train)), train=train))
        h = self.conv2(F.relu(self.bn2(h)))
        return F.relu(h + x)


class WBlock(chainer.Chain):
    def __init__(self, layer, in_size, out_size, stride=2):
        super(WBlock, self).__init__()
        links = [('a', WBottleNeckA(in_size, out_size, stride))]
        for i in range(layer-1):
            links += [('b{}'.format(i+1), WBottleNeckB(out_size,
                      out_size))]

        for l in links:
            self.add_link(*l)
        self.forward = links

    def __call__(self, x, train):
        for name, func in self.forward:
            x = func(x, train)
        return x


class WResNet(chainer.Chain):

    def __init__(self, window_len, feat_num):
        w = math.sqrt(2)
        self.window_len = window_len
        self.feat_num = feat_num
        self.train = True
        self.target_num = 1
        # w = normal.HeNormal(scale=1.0)
        super(WResNet, self).__init__(
            conv1=L.Convolution2D(1, 64, (feat_num, 7),
                                  1, (0, 3), w, nobias=True),
            res2=WBlock(3, 64, 64, 1),
            res3=WBlock(6, 64, 128, 2),
            res4=WBlock(3, 128, 256, 2),
            fc=MultiOut(256, 1, 3),
        )

    def precall(self, xs):
        xs = xs.reshape(xs.shape[0], self.window_len, -1)
        xs = xs.swapaxes(1, 2)
        xs = np.expand_dims(xs, 1)
        return xs

    def __call__(self, x):
        x = self.precall(x)
        h = self.conv1(x)
        h = F.max_pooling_2d(F.relu(h), (1, 3), stride=2)
        h = self.res2(h, self.train)
        h = self.res3(h, self.train)
        h = self.res4(h, self.train)
        h = F.average_pooling_2d(h, (1, 3), stride=1)
        h = self.fc(h)

        return h


class LSTM_DRN(chainer.Chain):

    def __init__(self, window_len, lstm_h, feat_num, n_out, out_size):
        w = math.sqrt(2)
        super(LSTM_DRN, self).__init__(
            conv1=L.Convolution2D(1, 64, (feat_num, 7),
                                  1, (0, 3), w, nobias=True),
            bn1=L.BatchNormalization(64),
            res2=Block(3, 64, 64, 1),
            res3=Block(6, 64, 128, 2),
            # res3_2=Block(6, 128, 256, 2),
            res4=Block(3, 128, 256, 2),
            lstm1=L.LSTM(256, lstm_h),
            out=MultiOut(lstm_h, n_out, out_size),
        )
        self.window_len = window_len
        self.feat_num = feat_num
        self.train = True
        self.target_num = n_out

    def reset_state(self):
        self.lstm1.reset_state()

    def __call__(self, x):
        # self.clear()
        h = self.conv1(x)
        # h = self.bn1(h, test=not self.train)
        h = F.max_pooling_2d(F.relu(h), (1, 3), stride=2)
        h = self.res2(h, self.train)
        h = self.res3(h, self.train)
        # h = self.res3_2(h, self.train)
        h = self.res4(h, self.train)
        h = F.average_pooling_2d(h, (1, 3), stride=1)
        # h = self.fc(h)
        h = self.lstm1(h)
        h = F.dropout(h, train=self.train)
        h = self.out(h)

        return h


if __name__ == "__main__":
    # net = LSTM_DRN(20, 50, 10, 1, 3)
    net = WResNet(20, 10)
    samples = np.random.random((32, 200)).astype(np.float32)
    loss = net(samples)
    print(loss[0].data)
