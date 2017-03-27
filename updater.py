"""
Custom updater of BackProp .
"""

from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import six

from chainer import training
from chainer import cuda
import cupy
logger = logging.getLogger(__name__)


class MLPUpdater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer, device, target_type):
        super(MLPUpdater, self).__init__(
            train_iter, optimizer, device=device)
        self.target_num = optimizer.target.predictor.target_num
        self.target_type = target_type

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Get the next batch (a list of tuples of two word IDs)
        batch = next(train_iter)
        logger.debug("batch: %s", batch)

        xs = batch[:, :-self.target_num]
        ts = batch[:, -self.target_num:]
        if self.target_type == 'c':
            ts = ts.astype(np.int32)

        loss = optimizer.target(xs, ts)
        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        optimizer.update()  # Update the parameters


DRNUpdater = MLPUpdater


class RollingWindowLSTMUpdater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer, device, target_type, window_len):
        super(RollingWindowLSTMUpdater, self).__init__(
            train_iter, optimizer, device=device)
        self.target_num = optimizer.target.predictor.target_num
        self.target_type = target_type
        self.window_len = window_len

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Get the next batch (a list of tuples of two word IDs)
        batch = next(train_iter)
        logger.debug("batch: %s", batch)

        xs = batch[:, :-self.target_num]
        xs = xs.reshape(xs.shape[0], self.window_len, -1)
        ts = batch[:, -self.target_num:]
        if 'c' in self.target_type:
            ts = ts.astype(np.int32)

        # Reset state at the begining.
        # Don't compute loss for previous window_len - 1 data.
        # NOTE: to fit 1 gpu training

        if self.device is not None and self.device >= 0:
            xs = cuda.to_gpu(xs)
            ts = cuda.to_gpu(ts)

        # NOTE: for pure LSTM
        optimizer.target.predictor.reset_state()
        for i in six.moves.xrange(self.window_len - 1):
            optimizer.target.predictor(xs[:, i, :])
        loss = optimizer.target(xs[:, -1, :], ts)

        # NOTE: for LSTMPlus
        # optimizer.target.predictor.reset_state()
        # for i in six.moves.xrange(self.window_len - 1):
        #     optimizer.target.predictor(xs[:, i, :10], xs[:, 0, 10:])
        # loss = optimizer.target(xs[:, -1, :10], xs[:, 0, 10:], ts)

        # NOTE: for rnn-cnn
        # xs = cupy.expand_dims(xs, axis=1)
        # loss = optimizer.target(xs, ts)
        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        optimizer.update()  # Update the parameters


class RollingWindowLSTMDRNUpdater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer, device, target_type,
                 window_len, slip_len):
        super(RollingWindowLSTMDRNUpdater, self).__init__(
            train_iter, optimizer, device=device)
        self.target_num = optimizer.target.predictor.target_num
        self.target_type = target_type
        self.window_len = window_len
        self.slip_len = slip_len

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Get the next batch (a list of tuples of two word IDs)
        batch = next(train_iter)
        logger.debug("batch: %s", batch)

        xs = batch[:, :-self.target_num]
        xs = xs.reshape(xs.shape[0], self.window_len+self.slip_len-1, -1)
        xs = xs.swapaxes(1, 2)
        xs = np.expand_dims(xs, 1)
        ts = batch[:, -self.target_num:]
        if self.target_type == 'c':
            ts = ts.astype(np.int32)


        # NOTE: for rnn-cnn
        # xs = cupy.expand_dims(xs, axis=1)
        # loss = optimizer.target(xs, ts)
        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        optimizer.update()  # Update the parameters


class RollingWindowLSTMDRNUpdater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer, device, target_type,
                 window_len, slip_len):
        super(RollingWindowLSTMDRNUpdater, self).__init__(
            train_iter, optimizer, device=device)
        self.target_num = optimizer.target.predictor.target_num
        self.target_type = target_type
        self.window_len = window_len
        self.slip_len = slip_len

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Get the next batch (a list of tuples of two word IDs)
        batch = next(train_iter)
        logger.debug("batch: %s", batch)

        xs = batch[:, :-self.target_num]
        xs = xs.reshape(xs.shape[0], self.window_len+self.slip_len-1, -1)
        xs = xs.swapaxes(1, 2)
        xs = np.expand_dims(xs, 1)
        ts = batch[:, -self.target_num:]
        if self.target_type == 'c':
            ts = ts.astype(np.int32)

        # Reset state at the begining.
        optimizer.target.predictor.reset_state()
        # Don't compute loss for previous window_len - 1 data.
        for i in six.moves.xrange(self.slip_len - 1):
            optimizer.target.predictor(xs[:, :, :, i:i+self.window_len])

        loss = optimizer.target(xs[:, :, :, self.slip_len-1:], ts)
        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        optimizer.update()  # Update the parameters


class DayBatchLSTMUpdater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer, device, target_type, bprop_len,
                 warmup_len):
        super(DayBatchLSTMUpdater, self).__init__(
            train_iter, optimizer, device=device)
        self.target_num = optimizer.target.predictor.target_num
        self.target_type = target_type
        self.bprop_len = bprop_len
        self.warmup_len = warmup_len

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Get the next batch (a list of tuples of two word IDs)
        batch = next(train_iter)
        logger.debug("batch: %s", batch)
        # NOTE(reed): We should sort the batch because of:
        # https://github.com/pfnet/chainer/issues/1183
        batch = sorted(batch, key=lambda b: b.shape[0], reverse=True)

        xs = [b[:, :-self.target_num] for b in batch]
        ts = [b[:, -self.target_num:] for b in batch]
        if self.target_type == 'c':
            ts = [t.astype(np.int32) for t in ts]
        total_lines = xs[0].shape[0]

        # Reset state at the begining.
        optimizer.target.predictor.reset_state()

        line = 0
        # Warm up.
        while line < self.warmup_len:
            batch_num = sum([1 for x in xs if x.shape[0] > line])
            x = np.stack([xs[j][line] for j in range(batch_num)])
            loss_list = optimizer.target.predictor(x)
            line += 1

        for loss in loss_list:
            loss.unchain_backward()

        # Progress the dataset iterator for bprop_len words at each iteration.
        while line < total_lines:
            loss = 0
            for k in six.moves.xrange(self.bprop_len):
                if line < total_lines:
                    batch_num = sum([1 for n in xs if n.shape[0] > line])
                    x = np.stack([xs[j][line] for j in range(batch_num)])
                    t = np.stack([ts[j][line] for j in range(batch_num)])

                    # Compute the loss at this time step and accumulate it
                    loss += optimizer.target(x, t)
                    line += 1
                else:
                    # NOTE(reed): make loss the same scale, hope this helps.
                    loss /= (float(k) / self.bprop_len)
                    break

            loss /= self.bprop_len
            optimizer.target.cleargrads()  # Clear the parameter gradients
            loss.backward()  # Backprop
            loss.unchain_backward()  # Truncate the graph
            optimizer.update()  # Update the parameters
