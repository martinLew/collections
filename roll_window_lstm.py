#!/usr/bin/env python
"""Sample script of neural network IF price model.

This code aims at apply lstm to IF data for predicting next price.
"""

from __future__ import division
from __future__ import print_function
import argparse
from functools import partial
import os
import re
import sys

import h5py
import numpy as np

import chainer
from chainer import optimizers, serializers
from chainer import training
from chainer.functions.evaluation import accuracy
from chainer.training import extensions

from data import load_gfund_data as load_data, rolling_window
from data import BASE_COLUMNS, KLINE_COLUMNS
from evaluator import RollingWindowLSTMEvaluator
from iterator import ArrayIterator
from model import LSTM, LSTMPlus
from mclassifier import MClassifier
from mregression import MRegression
from updater import RollingWindowLSTMUpdater

import logging
logging.basicConfig(
    format='%(asctime)s - %(process)d - %(filename)s(:%(lineno)d) '
           '[%(levelname)s] %(message)s')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--getdata', '-gd', type=bool, default=False)
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--freq', '-fq', default='1S',
                        help='frequency of sampling')
    parser.add_argument('--thresh', '-th', type=float, default=8e-5,
                        help='threshold of rise')
    parser.add_argument('--thresh1', '-th1', type=float, default=4e-5,
                        help='threshold of still')
    parser.add_argument('--window_len', '-w', type=int, default=30,
                        help='Number of lines for each data widow slice')
    parser.add_argument('--epoch', '-e', type=int, default=39,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gradclip', '-c', type=float, default=0,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--weight_decay', '-wd', type=float, default=0,
                        help='Weight decay at each step')
    parser.add_argument('--out', '-o', default='roll_window_lstm_result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit1', '-u1', type=int, default=50,
                        help='Number of units in each linear layer')
    parser.add_argument('--unit2', '-u2', type=int, default=200,
                        help='Number of units in each LSTM layer')

    parser.add_argument("--data_start_date", type=int, default=20140101,
                        help="data start date")
    parser.add_argument("--data_end_date", type=int, default=20140201,
                        help="data end date")
    parser.add_argument("--log_file", help="Log file name.")
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO", help="Log level.")

    parser.add_argument("--lr", type=float, default=1e-2,
                        help="Learning rate.")
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--shuffle", action='store_true', default=False,
                        help='shuffle training data')
    parser.add_argument("--sample_ratio", type=float, default=0.02)
    parser.add_argument("--target_steps", default="1",
                        help="target step list split by comma")
    parser.add_argument("--features", default="0,1,2,3,4,5,6,7,8,9",
                        help="feature list split by comma")
    parser.add_argument("--klines", default="",
                        help="klines list split by comma. "
                        "0 for last minute, 1 for last day, 2 for last week, "
                        "3 for last month.")
    parser.add_argument("--use_cache", action='store_true', default=False)
    parser.add_argument(
        "--target_type", choices=["c", "r", 'c1'], default="c",
        help="target type: c for classification, r for regression.")
    parser.add_argument("--batch_norm", action='store_true', default=True)
    parser.add_argument("--layer_norm", action='store_true', default=True)

    args = parser.parse_args()

    # Configure logging.
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file, mode='w')
        fmt = logging.Formatter(
            '%(asctime)s - %(process)d - %(filename)s(:%(lineno)d) '
            '[%(levelname)s] %(message)s')
        file_handler.setFormatter(fmt)
        logging.root.handlers = []
        logging.root.addHandler(file_handler)

    logger = logging.getLogger()
    logger.setLevel(args.log_level)

    # Set a lucky number as the seed: make the result reproducible.
    np.random.seed(28)

    # All base features used in the model.
    feat_cols = [int(f.strip()) for f in args.features.split(",") if f.strip()]
    # All targets used in the model.
    target_steps = [int(t.strip()) for t in args.target_steps.split(",")
                    if t.strip()]
    # target_cols = [t - 1 for t in target_steps]
    # modified by lym
    target_cols = range(len(target_steps))

    # All klines used in the model.
    kline_cols = [int(f.strip()) for f in args.klines.split(",") if f.strip()]
    # Each line has 5 columns: open, high, low, close, volume.
    kline_cols = [5 * i + j for i in kline_cols for j in range(5)]

    base_columns = BASE_COLUMNS[feat_cols].tolist()
    kline_columns = KLINE_COLUMNS[kline_cols].tolist()
    all_columns = np.concatenate([base_columns, kline_columns]).tolist()

    filename = 'dataset2_%s_%d_%d.h5' % (
        args.target_type, args.data_start_date, args.data_end_date)
    # Load IF data if exists, otherwise generate and save the file.
    if args.use_cache and os.path.isfile(filename):
        with h5py.File(filename, 'r') as file:
            days = file['days'].value
            states_list = [file['states_%d' % i][:] for i in range(days)]
            klines_list = [file['klines_%d' % i][:] for i in range(days)]
            targets_list = [file['targets_%d' % i][:] for i in range(days)]
    else:
        states_list, klines_list, targets_list = load_data(
             'IF', args.data_start_date, args.data_end_date,
             # target_steps=[1],
             target_steps=target_steps,
             target_type=args.target_type,
             freq=args.freq,
             rise_th=args.thresh,
             still_th=args.thresh1)
        days = len(states_list)
        with h5py.File(filename, 'w') as file:
            file.create_dataset('days', data=days)
            for i in range(days):
                file.create_dataset('states_%d' % i, data=states_list[i])
                file.create_dataset('klines_%d' % i, data=klines_list[i])
                file.create_dataset('targets_%d' % i, data=targets_list[i])

    # Keep only chosen columns & Make rolling data.
    states_list = [states[:, feat_cols] for states in states_list]

    # add by lym
    for astate in states_list:
        astate[:, 0:3] = (astate[:, 0:3]-astate[0, 0:3]) / astate[0, 0:3] * 40
        print(astate[1:3, 0:3])

    # Merge states and klines into states_list.
    if kline_cols:
        klines_list = [klines[:, kline_cols] for klines in klines_list]
        states_list = [np.concatenate(state_kline, axis=1)
                       for state_kline in zip(states_list, klines_list)]
    else:
        # Make it able to be freed
        klines_list = None

    states = np.concatenate([rolling_window(data, args.window_len)
                             for data in states_list])

    # added by lym
    # if kline_cols:
    #     klines = np.concatenate([rolling_window(data, args.window_len)
    #                             for data in klines_list])

    targets_list = [targets[:, target_cols] for targets in targets_list]
    targets = np.concatenate([rolling_window(target, args.window_len)[:, -1, :]
                              for target in targets_list])
    num_data = len(states)
    logger.info("Load %d data.", num_data)

    # NOTE(reed): Max-Min normalization. The first 3 columns are all
    # prices, so we use a unified max-min price, the others use per-column
    # max-min value to normalize.
    # Before normalization, we should trim some features: keep
    # feature values in the range [0.1%, 99.9%], and trim the outliers.
    # we don't trim the price features.
    state_re = states.reshape(-1, states.shape[-1])
    pat = re.compile(r'.*price.*|.*open$|.*high$|.*low$|.*close$')
    price_cols = [i for i in range(len(all_columns))
                  if pat.findall(all_columns[i])]
    other_cols = set(range(len(all_columns))) - set(price_cols)
    if 'serial_id' in all_columns:
        other_cols -= {all_columns.index('serial_id')}
    other_cols = list(other_cols)
    logger.info("all_columns: %s", all_columns)
    logger.info("price_cols: %s", price_cols)
    logger.info("other_cols: %s", other_cols)
    for i in other_cols:
        features = state_re[:, i]
        low = np.percentile(features, 0.1)
        high = np.percentile(features, 99.9)
        features[features < low] = low
        features[features > high] = high

    if args.getdata:
        with open('min_ptp.dat', 'w') as fp:
            for i in range(state_re[0, :].size):
                fp.write(str(state_re[:, i].min()) + ' ' +
                         str(state_re[:, i].ptp()) + '\n')
        sys.exit(2)

    # price_min = state_re[:, price_cols].min()
    # price_ptp = state_re[:, price_cols].ptp()
    # state_re[:, price_cols] = (
    #     (state_re[:, price_cols] - price_min) / price_ptp)
    state_re[:, other_cols] = (
        (state_re[:, other_cols] - state_re[:, other_cols].min(0))
        / state_re[:, other_cols].ptp(0))

    # Sampling rows.
    logger.info("Sampling data...")
    # NOTE(reed): rows which have non -1 target(s) - rows with all targets as
    # -1 are useless so no need to keep.
    valid_rows = np.any(targets != -1, axis=1)

    # Positive and negative rows are not sampled.
    if 'c' in args.target_type:
        positive_rows = np.any(targets == 0, axis=1)
        negative_rows = np.any(targets == 2, axis=1)
    else:
        positive_rows = np.any(targets > 0.0004, axis=1)
        negative_rows = np.any(targets < -0.0004, axis=1)

    # Sampled random rows according to the sample ratio.
    random_rows = (np.random.uniform(0, 1, (targets.shape[0], ))
                   < args.sample_ratio)

    # All positive rows, negative rows and sampled rows - which contains
    # invalid rows with all their targets as -1.
    sample_rows = np.any(
        np.stack([random_rows, positive_rows, negative_rows], axis=1), axis=1)

    # Get rid of invalid rows with all their targets as -1.
    keep_rows = np.all(np.stack([sample_rows, valid_rows], axis=1), axis=1)

    states = states[keep_rows]
    targets = targets[keep_rows]

    if 'c' in args.target_type:
        for i, t in enumerate(target_steps):
            all_targets = targets[:, i].reshape(-1) + 1
            all_target_bincount = np.bincount(all_targets.astype(np.int32))
            index = np.nonzero(all_target_bincount)[0]
            all_target_freq = zip(index - 1, all_target_bincount[index])

            logger.info("all_target_step_%d_freq: %s", t, all_target_freq)
    else:
        for i, t in enumerate(target_steps):
            all_targets = targets[:, i].reshape(-1)
            hist, bin_edges = np.histogram(
                all_targets, bins=[-0.01, -0.0004, 0, 0.0004, 0.01])
            all_target_freq = zip(bin_edges[:-1], bin_edges[1:], hist)

            logger.info("all_target_step_%d_freq: %s", t, all_target_freq)

    states = states.reshape(states.shape[0], -1)

    all_data = np.concatenate([states, targets], axis=1)
    num_data = len(all_data)
    # NOTE(reed): FOR STARTING, Shuffle the data to exclude the trend influence
    # np.random.shuffle(all_data)
    train = all_data[:num_data // 5 * 4]
    valid = all_data[num_data // 5 * 4:]

    if 'c' in args.target_type:
        for i, t in enumerate(target_steps):
            train_targets = train[:, i - len(target_steps)].reshape(-1) + 1
            train_target_bincount = np.bincount(train_targets.astype(np.int32))
            index = np.nonzero(train_target_bincount)[0]
            train_target_freq = zip(index - 1, train_target_bincount[index])

            valid_targets = valid[:, i - len(target_steps)].reshape(-1) + 1
            valid_target_bincount = np.bincount(valid_targets.astype(np.int32))
            valid_target_freq = zip(index - 1, valid_target_bincount[index])

            logger.info("train_target_step_%d_freq: %s", t, train_target_freq)
            logger.info("valid_target_step_%d_freq: %s", t, valid_target_freq)
    else:
        for i, t in enumerate(target_steps):
            train_targets = train[:, i - len(target_steps)].reshape(-1)
            hist, bin_edges = np.histogram(
                train_targets, bins=[-0.1, -0.001, 0, 0.001, 0.1])
            train_target_freq = zip(bin_edges[:-1], bin_edges[1:], hist)

            valid_targets = valid[:, i - len(target_steps)].reshape(-1)
            hist, bin_edges = np.histogram(
                valid_targets, bins=[-0.01, -0.0004, 0, 0.0004, 0.01])
            valid_target_freq = zip(bin_edges[:-1], bin_edges[1:], hist)

            logger.info("train_target_step_%d_freq: %s", t, train_target_freq)
            logger.info("valid_target_step_%d_freq: %s", t, valid_target_freq)

    # Data iterator.
    train_iter = ArrayIterator(train, args.batchsize, shuffle=args.shuffle)
    valid_iter = ArrayIterator(valid, args.batchsize, repeat=False)

    # Prepare an LSTM model
    out_size = 11 if 'c' in args.target_type else 1
    # NOTE: LSTM input size should be 10, because klines added at tail
    # states.shape[-1] // args.window_len
    lstm = LSTM(states.shape[-1] // args.window_len, args.unit1, args.unit2,
                    n_out=len(target_cols), out_size=out_size,
                    batch_norm=args.batch_norm, layer_norm=args.layer_norm)
    # lstm = CNNTradeModel(([5, 10, 15], [1, 1, 1]), 1, 3)
    if 'c' in args.target_type:
        # Set -1 as the ignore lable when computing accuracy.
        accfun = partial(accuracy.accuracy, ignore_label=-1)
        model = MClassifier(lstm, accfun=accfun)
    else:
        model = MRegression(lstm)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # make the GPU current
        model.to_gpu()

    # Model with shared params and distinct states
    eval_model = model.copy()
    eval_mlp = eval_model.predictor
    eval_mlp.train = False

    # Set up an optimizer
    optimizer = optimizers.Adam(alpha=args.lr, beta1=args.alpha,
                                beta2=args.beta, eps=args.eps)
    optimizer.setup(model)
    if args.gradclip:
        optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))
    if args.weight_decay:
        optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

    # Set up a trainer
    updater = RollingWindowLSTMUpdater(train_iter, optimizer, args.gpu,
                                       args.target_type, args.window_len)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(RollingWindowLSTMEvaluator(
        valid_iter, eval_model, device=args.gpu, target_type=args.target_type,
        window_len=args.window_len))

    trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
    print_fields = ['epoch', 'iteration']
    for i in range(len(target_steps)):
        print_fields.extend(['main/loss_%d' % i,
                             'validation/main/loss_%d' % i])
        if 'c' in args.target_type:
            print_fields.extend(['main/accuracy_%d' % i,
                                 'validation/main/accuracy_%d' % i])
    trainer.extend(extensions.PrintReport(print_fields), trigger=(1, 'epoch'))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.snapshot(
        filename='snapshot_epoch_{.updater.epoch}'))
    trainer.extend(extensions.snapshot_object(
        model, 'model_epoch_{.updater.epoch}'))
    if args.resume:
        serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
