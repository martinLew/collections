import copy
from collections import Counter
import os

import six

import numpy as np
import cupy

from chainer.dataset import convert
from chainer import reporter as reporter_module
from chainer.training import extension, extensions
from chainer import cuda


# Custom evaluator for mlp.
class MLPEvaluator(extensions.Evaluator):
    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = extension.PRIORITY_WRITER

    def __init__(self, iterator, target, converter=convert.concat_examples,
                 device=None, eval_hook=None, eval_func=None, target_type='c'):
        super(MLPEvaluator, self).__init__(
            iterator, target, converter, device, eval_hook, eval_func)
        self.target_num = target.predictor.target_num
        self.target_type = target_type
        if self.target_type == 'c':
            if os.path.isfile('confusion_matrix_roll_window.txt'):
                f = open('confusion_matrix_roll_window.txt', 'w')
                f.close()
            if os.path.isfile('bin_confusion_matrix_roll_window.txt'):
                f = open('bin_confusion_matrix_roll_window.txt', 'w')
                f.close()

    def evaluate(self):
        """Evaluates the model and returns a result dictionary.

        This method runs the evaluation loop over the validation dataset. It
        accumulates the reported values to :class:`~chainer.DictSummary` and
        returns a dictionary whose values are means computed by the summary.

        Users can override this method to customize the evaluation routine.

        Returns:
            dict: Result dictionary. This dictionary is further reported via
                :func:`~chainer.report` without specifying any observer.

        """
        iterator = self._iterators['main']
        target = self._targets['main']
        eval_func = self.eval_func or target

        if self.eval_hook:
            self.eval_hook(self)
        it = copy.copy(iterator)
        summary = reporter_module.DictSummary()

        outputs = []
        bin_outputs = []
        for i in six.moves.xrange(self.target_num):
            outputs.append([])
            bin_outputs.append([])

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                xs = batch[:, :-self.target_num]
                ts = batch[:, -self.target_num:]
                if self.target_type == 'c':
                    ts = ts.astype(np.int32)

                eval_func(xs, ts)

                # Calculate confusion matrix.
                if self.target_type == 'c':
                    for i in six.moves.xrange(self.target_num):
                        pred = np.argmax(target.y[i].data, axis=1)
                        outputs[i].extend(zip(ts[:, i], pred))

                        bin_pred = np.argmax(target.y[i].data[:, (0, 2)],
                                             axis=1)
                        bin_outputs[i].extend(zip(ts[:, i], bin_pred))

            summary.add(observation)

        if self.target_type == 'c':
            with open('confusion_matrix_roll_window.txt', 'a') as f:
                six.print_("===================", file=f)
                for i in six.moves.xrange(self.target_num):
                    nclass = target.y[i].data.shape[1]
                    counter = Counter(outputs[i])
                    # confusion matrix
                    cm = np.zeros((nclass+1, nclass+1), dtype=np.int32)
                    for c in counter:
                        cm[c] = counter[c]
                    six.print_("*******************", file=f)
                    six.print_(cm, file=f)

            with open('bin_confusion_matrix_roll_window.txt', 'a') as f:
                six.print_("===================", file=f)
                for i in six.moves.xrange(self.target_num):
                    nclass = target.y[i].data.shape[1]
                    counter = Counter(bin_outputs[i])
                    # NOTE(reed): binary confusion matrix - which just predict
                    # the target to be positive or negative, not neutral.
                    cm = np.zeros((nclass+1, nclass), dtype=np.int32)
                    for c in counter:
                        cm[c] = counter[c]
                    six.print_("*******************", file=f)
                    six.print_(cm, file=f)

        return summary.compute_mean()

DRNEvaluator = MLPEvaluator


# Custom evaluator for rolling window lstm.
class RollingWindowLSTMEvaluator(extensions.Evaluator):
    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = extension.PRIORITY_WRITER

    def __init__(self, iterator, target, converter=convert.concat_examples,
                 device=None, eval_hook=None, eval_func=None, target_type='c',
                 window_len=10):
        super(RollingWindowLSTMEvaluator, self).__init__(
            iterator, target, converter, device, eval_hook, eval_func)
        self.target_num = target.predictor.target_num
        self.target_type = target_type
        self.window_len = window_len
        if 'c' in self.target_type:
            if os.path.isfile('confusion_matrix_roll_window.txt'):
                f = open('confusion_matrix_roll_window.txt', 'w')
                f.close()
            if os.path.isfile('bin_confusion_matrix_roll_window.txt'):
                f = open('bin_confusion_matrix_roll_window.txt', 'w')
                f.close()

    def evaluate(self):
        """Evaluates the model and returns a result dictionary.

        This method runs the evaluation loop over the validation dataset. It
        accumulates the reported values to :class:`~chainer.DictSummary` and
        returns a dictionary whose values are means computed by the summary.

        Users can override this method to customize the evaluation routine.

        Returns:
            dict: Result dictionary. This dictionary is further reported via
                :func:`~chainer.report` without specifying any observer.

        """
        iterator = self._iterators['main']
        target = self._targets['main']
        eval_func = self.eval_func or target

        if self.eval_hook:
            self.eval_hook(self)
        it = copy.copy(iterator)
        summary = reporter_module.DictSummary()

        outputs = []
        bin_outputs = []
        for i in six.moves.xrange(self.target_num):
            outputs.append([])
            bin_outputs.append([])

        for batch in it:
            # added by lym
            if self.device is not None and self.device >= 0:
                batch = cuda.to_gpu(batch)
            observation = {}
            with reporter_module.report_scope(observation):
                xs = batch[:, :-self.target_num]
                xs = xs.reshape(xs.shape[0], self.window_len, -1)
                ts = batch[:, -self.target_num:]
                if 'c' in self.target_type:
                    ts = ts.astype(np.int32)

                # Reset state at the begining.
                # Don't compute loss for previous window_len - 1 data.
                # NOTE: for classical LSTM
                target.predictor.reset_state()
                for i in six.moves.xrange(self.window_len-1):
                    target.predictor(xs[:, i, :])
                eval_func(xs[:, -1, :], ts)

                # NOTE: for LSTMPlus
                # target.predictor.reset_state()
                # for i in six.moves.xrange(self.window_len-1):
                #     target.predictor(xs[:, i, :10], xs[:, 0, 10:])
                # eval_func(xs[:, -1, :10], xs[:, 0, 10:], ts)

                # NOTE: for CNN
                # xs = cupy.expand_dims(xs, axis=1)
                # eval_func(xs, ts)

                # Calculate confusion matrix.
                if 'c' in self.target_type:
                    for i in six.moves.xrange(self.target_num):
                        pred = np.argmax(cuda.to_cpu(target.y[i].data), axis=1)
                        outputs[i].extend(zip(cuda.to_cpu(ts[:, i]), pred))

                        bin_pred = np.argmax(cuda.to_cpu(target.y[i].data[:, [0, 2]]),
                                             axis=1)
                        bin_outputs[i].extend(zip(cuda.to_cpu(ts[:, i]), bin_pred))

            summary.add(observation)

        if 'c' in self.target_type:
            with open('confusion_matrix_roll_window.txt', 'a') as f:
                six.print_("===================", file=f)
                for i in six.moves.xrange(self.target_num):
                    nclass = target.y[i].data.shape[1]
                    counter = Counter(outputs[i])
                    cm = np.zeros((nclass+1, nclass+1), dtype=np.int32)
                    for c in counter:
                        cm[c] = counter[c]
                    six.print_("*******************", file=f)
                    six.print_(cm, file=f)

            # with open('bin_confusion_matrix_roll_window.txt', 'a') as f:
            #     six.print_("===================", file=f)
            #     for i in six.moves.xrange(self.target_num):
            #         nclass = target.y[i].data.shape[1]
            #         counter = Counter(bin_outputs[i])
            #         # NOTE(reed): binary confusion matrix - which just predict
            #         # the target to be positive or negative, not neutral.
            #         cm = np.zeros((nclass+1, nclass), dtype=np.int32)
            #         for c in counter:
            #             cm[c] = counter[c]
            #         six.print_("*******************", file=f)
            #         six.print_(cm, file=f)

        return summary.compute_mean()


# Custom evaluator for rolling window lstm.
class RollingWindowLSTMDRNEvaluator(extensions.Evaluator):
    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = extension.PRIORITY_WRITER

    def __init__(self, iterator, target, converter=convert.concat_examples,
                 device=None, eval_hook=None, eval_func=None, target_type='c',
                 window_len=10, slip_len=20):
        super(RollingWindowLSTMDRNEvaluator, self).__init__(
            iterator, target, converter, device, eval_hook, eval_func)
        self.target_num = target.predictor.target_num
        self.target_type = target_type
        self.window_len = window_len
        self.slip_len = slip_len
        if self.target_type == 'c':
            if os.path.isfile('confusion_matrix_roll_window.txt'):
                f = open('confusion_matrix_roll_window.txt', 'w')
                f.close()
            if os.path.isfile('bin_confusion_matrix_roll_window.txt'):
                f = open('bin_confusion_matrix_roll_window.txt', 'w')
                f.close()

    def evaluate(self):
        """Evaluates the model and returns a result dictionary.

        This method runs the evaluation loop over the validation dataset. It
        accumulates the reported values to :class:`~chainer.DictSummary` and
        returns a dictionary whose values are means computed by the summary.

        Users can override this method to customize the evaluation routine.

        Returns:
            dict: Result dictionary. This dictionary is further reported via
                :func:`~chainer.report` without specifying any observer.

        """
        iterator = self._iterators['main']
        target = self._targets['main']
        eval_func = self.eval_func or target

        if self.eval_hook:
            self.eval_hook(self)
        it = copy.copy(iterator)
        summary = reporter_module.DictSummary()

        outputs = []
        bin_outputs = []
        for i in six.moves.xrange(self.target_num):
            outputs.append([])
            bin_outputs.append([])

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                xs = batch[:, :-self.target_num]
                xs = xs.reshape(xs.shape[0], self.window_len+self.slip_len-1,
                                -1)
                xs = xs.swapaxes(1, 2)
                xs = np.expand_dims(xs, 1)
                ts = batch[:, -self.target_num:]
                if self.target_type == 'c':
                    ts = ts.astype(np.int32)

                # Reset state at the begining.
                target.predictor.reset_state()
                # Don't compute loss for previous window_len - 1 data.
                for i in six.moves.xrange(self.slip_len-1):
                    target.predictor(xs[:, :, :, i:i+self.window_len])

                eval_func(xs[:, :, :, self.slip_len-1:], ts)

                # Calculate confusion matrix.
                if self.target_type == 'c':
                    for i in six.moves.xrange(self.target_num):
                        pred = np.argmax(target.y[i].data, axis=1)
                        outputs[i].extend(zip(ts[:, i], pred))

                        bin_pred = np.argmax(target.y[i].data[:, (0, 2)],
                                             axis=1)
                        bin_outputs[i].extend(zip(ts[:, i], bin_pred))

            summary.add(observation)

        if self.target_type == 'c':
            with open('confusion_matrix_roll_window.txt', 'a') as f:
                six.print_("===================", file=f)
                for i in six.moves.xrange(self.target_num):
                    nclass = target.y[i].data.shape[1]
                    counter = Counter(outputs[i])
                    # confusion matrix
                    cm = np.zeros((nclass+1, nclass+1), dtype=np.int32)
                    for c in counter:
                        cm[c] = counter[c]
                    six.print_("*******************", file=f)
                    six.print_(cm, file=f)

            with open('bin_confusion_matrix_roll_window.txt', 'a') as f:
                six.print_("===================", file=f)
                for i in six.moves.xrange(self.target_num):
                    nclass = target.y[i].data.shape[1]
                    counter = Counter(bin_outputs[i])
                    # NOTE(reed): binary confusion matrix - which just predict
                    # the target to be positive or negative, not neutral.
                    cm = np.zeros((nclass+1, nclass), dtype=np.int32)
                    for c in counter:
                        cm[c] = counter[c]
                    six.print_("*******************", file=f)
                    six.print_(cm, file=f)

        return summary.compute_mean()


# Custom evaluator for day batch lstm.
class DayBatchLSTMEvaluator(extensions.Evaluator):
    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = extension.PRIORITY_WRITER

    def __init__(self, iterator, target, converter=convert.concat_examples,
                 device=None, eval_hook=None, eval_func=None, target_type='c',
                 warmup_len=600):
        super(DayBatchLSTMEvaluator, self).__init__(
            iterator, target, converter, device, eval_hook, eval_func)
        self.target_num = target.predictor.target_num
        self.target_type = target_type
        self.warmup_len = warmup_len
        if self.target_type == 'c':
            if os.path.isfile('confusion_matrix_roll_window.txt'):
                f = open('confusion_matrix_roll_window.txt', 'w')
                f.close()
            if os.path.isfile('bin_confusion_matrix_roll_window.txt'):
                f = open('bin_confusion_matrix_roll_window.txt', 'w')
                f.close()

    def evaluate(self):
        """Evaluates the model and returns a result dictionary.

        This method runs the evaluation loop over the validation dataset. It
        accumulates the reported values to :class:`~chainer.DictSummary` and
        returns a dictionary whose values are means computed by the summary.

        Users can override this method to customize the evaluation routine.

        Returns:
            dict: Result dictionary. This dictionary is further reported via
                :func:`~chainer.report` without specifying any observer.

        """
        iterator = self._iterators['main']
        target = self._targets['main']
        eval_func = self.eval_func or target

        if self.eval_hook:
            self.eval_hook(self)
        it = copy.copy(iterator)
        summary = reporter_module.DictSummary()

        outputs = []
        bin_outputs = []
        for i in six.moves.xrange(self.target_num):
            outputs.append([])
            bin_outputs.append([])

        for batch in it:
            # NOTE(reed): We should sort the batch because of:
            # https://github.com/pfnet/chainer/issues/1183
            batch = sorted(batch, key=lambda b: b.shape[0], reverse=True)
            xs = [b[:, :-self.target_num] for b in batch]
            ts = [b[:, -self.target_num:] for b in batch]
            if self.target_type == 'c':
                ts = [t.astype(np.int32) for t in ts]
            total_lines = xs[0].shape[0]

            # Reset state at the begining.
            target.predictor.reset_state()
            # Warm up.
            for i in six.moves.xrange(self.warmup_len):
                batch_num = sum([1 for x in xs if x.shape[0] > i])
                x = np.stack([xs[j][i] for j in range(batch_num)])
                target.predictor(x)

            for i in six.moves.xrange(self.warmup_len, total_lines):
                observation = {}
                with reporter_module.report_scope(observation):
                    batch_num = sum([1 for n in xs if n.shape[0] > i])
                    x = np.stack([xs[j][i] for j in range(batch_num)])
                    t = np.stack([ts[j][i] for j in range(batch_num)])
                    eval_func(x, t)
                    # Calculate confusion matrix.
                    if self.target_type == 'c':
                        for j in six.moves.xrange(self.target_num):
                            pred = np.argmax(target.y[j].data, axis=1)
                            outputs[j].extend(zip(t[:, j], pred))

                            bin_pred = np.argmax(target.y[j].data[:, (0, 2)],
                                                 axis=1)
                            bin_outputs[j].extend(zip(t[:, j], bin_pred))
                summary.add(observation)

        if self.target_type == 'c':
            with open('confusion_matrix_roll_window.txt', 'a') as f:
                six.print_("===================", file=f)
                for i in six.moves.xrange(self.target_num):
                    nclass = target.y[i].data.shape[1]
                    counter = Counter(outputs[i])
                    # confusion matrix
                    cm = np.zeros((nclass+1, nclass+1), dtype=np.int32)
                    for c in counter:
                        cm[c] = counter[c]
                    six.print_("*******************", file=f)
                    six.print_(cm, file=f)

            with open('bin_confusion_matrix_roll_window.txt', 'a') as f:
                six.print_("===================", file=f)
                for i in six.moves.xrange(self.target_num):
                    nclass = target.y[i].data.shape[1]
                    counter = Counter(bin_outputs[i])
                    # NOTE(reed): binary confusion matrix - which just predict
                    # the target to be positive or negative, not neutral.
                    cm = np.zeros((nclass+1, nclass), dtype=np.int32)
                    for c in counter:
                        cm[c] = counter[c]
                    six.print_("*******************", file=f)
                    six.print_(cm, file=f)

        return summary.compute_mean()
