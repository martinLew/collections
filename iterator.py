
from __future__ import division
from __future__ import print_function

import logging
import numpy as np

import chainer

logger = logging.getLogger(__name__)


# Dataset iterator to create a batch of sequences at different positions.
class ArrayIterator(chainer.dataset.Iterator):

    def __init__(self, dataset, batch_size, repeat=True, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size  # batch size
        # Number of completed sweeps over the dataset. In this case, it is
        # incremented if every day data is visited at least once after the last
        # increment.
        self.epoch = 0
        # True if the epoch is incremented at the last iteration.
        self.is_new_epoch = False
        self.repeat = repeat
        self.shuffle = shuffle
        self.length = len(dataset)
        # Offsets maintain the position of each sequence in the mini-batch.
        self.offsets = [i * self.length // batch_size
                        for i in range(batch_size)]
        # NOTE: this is not a count of parameter updates. It is just a count of
        # calls of ``__next__``.
        self.iteration = 0
        if self.shuffle:
            np.random.shuffle(self.dataset)

    def __next__(self):
        # This iterator returns a list representing a mini-batch. Each item
        # indicates a different day in the original sequence.
        # At each iteration, the iteration count is incremented, which pushes
        # forward the "current" position.
        if not self.repeat and self.iteration * self.batch_size >= self.length:
            # If not self.repeat, this iterator stops at the end of the first
            # epoch (i.e., when all words are visited once).
            raise StopIteration

        cur_data = self.get_data()
        self.iteration += 1

        epoch = self.iteration * self.batch_size // self.length
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch
            if self.shuffle:
                np.random.shuffle(self.dataset)

        return cur_data

    @property
    def epoch_detail(self):
        # Floating point version of epoch.
        return self.iteration * self.batch_size / self.length

    def get_data(self):
        # It returns the current data.
        return np.stack(
            [self.dataset[(offset + self.iteration) % self.length]
             for offset in self.offsets])

    def serialize(self, serializer):
        # It is important to serialize the state to be recovered on resume.
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)


# Dataset iterator to create a batch of sequences at different positions.
class ArrayListIterator(chainer.dataset.Iterator):

    def __init__(self, dataset, batch_size, repeat=True, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size  # batch size
        # Number of completed sweeps over the dataset. In this case, it is
        # incremented if every day data is visited at least once after the last
        # increment.
        self.epoch = 0
        # True if the epoch is incremented at the last iteration.
        self.is_new_epoch = False
        self.repeat = repeat
        self.shuffle = shuffle
        self.length = len(dataset)
        # Offsets maintain the position of each sequence in the mini-batch.
        self.offsets = [i * self.length // batch_size
                        for i in range(batch_size)]
        # NOTE: this is not a count of parameter updates. It is just a count of
        # calls of ``__next__``.
        self.iteration = 0
        if self.shuffle:
            np.random.shuffle(self.dataset)

    def __next__(self):
        # This iterator returns a list representing a mini-batch. Each item
        # indicates a different day in the original sequence.
        # At each iteration, the iteration count is incremented, which pushes
        # forward the "current" position.
        if not self.repeat and self.iteration * self.batch_size >= self.length:
            # If not self.repeat, this iterator stops at the end of the first
            # epoch (i.e., when all words are visited once).
            raise StopIteration

        cur_data = self.get_data()
        self.iteration += 1

        epoch = self.iteration * self.batch_size // self.length
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch
            if self.shuffle:
                np.random.shuffle(self.dataset)

        return cur_data

    @property
    def epoch_detail(self):
        # Floating point version of epoch.
        return self.iteration * self.batch_size / self.length

    def get_data(self):
        # It returns the current data.
        return [self.dataset[(offset + self.iteration) % self.length]
                for offset in self.offsets]

    def serialize(self, serializer):
        # It is important to serialize the state to be recovered on resume.
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)
