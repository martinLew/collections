# coding: utf-8
import argparse
from datetime import datetime
import math
import logging
import h5py
import multiprocessing as mp
import numpy as np
import pandas as pd
import pymongo
import cPickle as cp
import Indicator

from gfunddataprocessor import gfunddataprocessor as gfd

recordFlag = True
recFile = open('trigger.dat', 'w')

logging.basicConfig(
    format='%(asctime)s - %(process)d - %(filename)s(:%(lineno)d) '
           '[%(levelname)s] %(message)s')


logger = logging.getLogger(__name__)

# Feature number.
BASE_TIME = datetime(1970, 1, 1)
COLUMNS1 = np.array(['last_price', 'avg_price', 'mid_price', 'volume',
                     'open_interest_diff', 'volume_direction', 'volume_depth',
                     'ups', 'downs', 'spread', 'accu_volume', 'accu_turnover',
                     'open_interest', 'ask_price', 'bid_price', 'ask_volume',
                     'bid_volume', 'serial_id'])
FEAT_NUM1 = 18

BASE_COLUMNS = np.array(['open', 'close', 'mid_price', 'volume',
                         'open_interest_diff', 'volume_direction',
                         'volume_depth', 'ups', 'downs', 'spread',
                         'spread2', 'mean_lastprice',
                         'mean_askprice1', 'mean_askvolume1',
                         'mean_bidprice1', 'mean_bidvolume1',
                         'mean_volume', 'mean_diff_interest',
                         'volatility',
                         'bid_price1', 'bid_volume1', 'ask_price1',
                         'ask_volume1', 'serial_id'])

KLINE_COLUMNS = np.array(
    ['last_minute_open', 'last_minute_high', 'last_minute_low',
     'last_minute_close', 'last_minute_volume',
     'last_15_minute_open', 'last_15_minute_high', 'last_15_minute_low',
     'last_15_minute_close', 'last_15_minute_volume',
     'last_day_open', 'last_day_high', 'last_day_low',
     'last_day_close', 'last_day_volume',
     'last_week_open', 'last_week_high', 'last_week_low',
     'last_week_close', 'last_week_volume'])

ALL_COLUMNS = np.concatenate([BASE_COLUMNS, KLINE_COLUMNS])


def rolling_window(a, lag):
    """
    Convert a into time-lagged vectors - forked from neon

    a    : (n, p)
    lag  : time steps used for prediction

    returns  (n-lag+1, lag, p)  array
    """
    assert a.shape[0] > lag

    shape = [a.shape[0] - lag + 1, lag, a.shape[-1]]
    strides = [a.strides[0], a.strides[0], a.strides[-1]]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def load_history_data(
        start_date, end_date=None, host='172.16.80.29', port=27017,
        db_name='tick_data_2010-2015', col='col1', target_steps=[1, 2],
        target_type='c'):
    """Load history raw data from mongodb"""

    dbClient = pymongo.MongoClient(host, port)
    collection = dbClient[db_name][col]

    # Set the date range.
    if not end_date:
        flt = {'DateValue': {'$gte': start_date}}
    else:
        flt = {'DateValue': {'$gte': start_date,
                             '$lte': end_date}}
    db_cursor = collection.find(flt)

    states_list = []
    targets_list = []
    date = None
    tmp_data = []
    max_target = max(target_steps)

    def _strip_features(day_datas):
        """Strip features from market data within a day."""
        length = len(day_datas)
        day_datas = sorted(day_datas, key=lambda d: d['timestamp'])
        state = np.zeros((length - max_target - 1, FEAT_NUM1),
                         dtype=np.float32)
        target = np.empty((length - max_target - 1, len(target_steps)),
                          dtype=np.float32)

        # NOTE(reed): In softmax_cross_entropy, it takes -1 as ignore label.
        target[:] = -1

        valid = True
        for j in range(1, length - max_target - 1):
            this_data = day_datas[j]
            last_data = day_datas[j - 1]
            next_data = day_datas[j + 1]
            # State
            #  0: last_price
            #  1: avg_price - the original "turnover" is accumulated.
            #  2: mid_price
            #  3: volume - the original "volume" is accumulated.
            #  4: open_interest_diff
            #  5: volume_direction
            #  6: volume_depth
            #  7: ups - tick number continuously goes up
            #  8: downs - tick number continuously goes down
            #  9: spread
            # 10: accu_volume
            # 11: accu_turnover
            # 12: open_interest
            # 13: ask_price
            # 14: bid_price
            # 15: ask_volume
            # 16: bid_volume
            # 17: serial_id
            try:
                state[j, 0] = this_data['last_price']

                volume = this_data['volume'] - last_data['volume']
                if volume > 0:
                    # NOTE(reed): the contract size of IF is 300
                    state[j, 1] = (this_data['turnover']
                                   - last_data['turnover']) / volume / 300
                # If last avg_price exists, use it.
                elif state[j-1, 1]:
                    state[j, 1] = state[j-1, 1]
                # Else use last_price instead.
                else:
                    state[j, 1] = state[j, 0]

                state[j, 2] = (this_data['ask_price1']
                               + this_data['bid_price1']) / 2

                state[j, 3] = volume

                state[j, 4] = (this_data['open_interest']
                               - last_data['open_interest'])

                state[j, 5] = math.log(
                    this_data['ask_volume1'] / this_data['bid_volume1'])

                state[j, 6] = (this_data['ask_volume1']
                               + this_data['bid_volume1'])

                # NOTE(reed): Since state.dtype is float32, and the minimal
                # price change of IF is 0.2, we use 0.1 here to prevent from
                # floating error.
                if state[j, 0] > state[j-1, 0] + 0.0001:
                    state[j, 7] = state[j-1, 7] + 1
                    state[j, 8] = 0
                elif state[j, 0] < state[j-1, 0] - 0.0001:
                    state[j, 7] = 0
                    state[j, 8] = state[j-1, 8] + 1
                else:
                    state[j, 7] = state[j-1, 7]
                    state[j, 8] = state[j-1, 8]

                state[j, 9] = this_data['ask_price1'] - this_data['bid_price1']

                state[j, 10] = this_data['volume']
                state[j, 11] = this_data['turnover']
                state[j, 12] = this_data['open_interest']
                state[j, 13] = this_data['ask_price1']
                state[j, 14] = this_data['bid_price1']
                state[j, 15] = this_data['ask_volume1']
                state[j, 16] = this_data['bid_volume1']

                # target
                next_price = next_data['last_price']
                if target_type == 'c':  # classification
                    for k, t in enumerate(target_steps):
                        target_price = day_datas[j+t+1]['last_price']
                        ratio = target_price / next_price - 1
                        if ratio > 0.0002:
                            target[j, k] = 0
                        elif ratio < -0.0002:
                            target[j, k] = 2
                        elif -0.00005 < ratio < 0.00005:
                            target[j, k] = 1
                        target[j, k] = ratio
                        # NOTE(reed): else the target is left to be -1 as
                        # ignored.
                else:  # regression
                    y = np.empty((max_target+1, ), dtype=np.float64)
                    A = np.ones((max_target+1, 2), dtype=np.float64)
                    for t in range(max_target+1):
                        y[t] = day_datas[j+1+t]['last_price'] / next_price
                        A[t, 0] = day_datas[j+1+t]['timestamp']
                    # Use the first timestamp as original point.
                    A[:, 0] -= A[0, 0]

                    for k, t in enumerate(target_steps):
                        m, _ = np.linalg.lstsq(A[:t+1], y[:t+1])[0]
                        target[j, k] = m

            except Exception, e:
                logger.warn(e)
                valid = False
                break

        if valid:
            state = np.delete(state, (0, ), axis=0)
            target = np.delete(target, (0, ), axis=0)

        return (state, target) if valid else None

    for i, d in enumerate(db_cursor):
        last_date = date
        date = d['DateValue']

        if i > 0 and date != last_date:
            logger.info("last_date: %d, date: %d", last_date, date)
            ret = _strip_features(tmp_data)
            if ret:
                states_list.append(ret[0])
                targets_list.append(ret[1])

            del tmp_data[:]

        volume = float(d['Volume'])
        if volume:
            data = {}
            timestamp = d['timestamp']
            form = ("%Y-%m-%dT%H:%M:%S.%f" if len(timestamp) > 19
                    else "%Y-%m-%dT%H:%M:%S")
            time = datetime.strptime(timestamp, form)
            data['timestamp'] = (time - BASE_TIME).total_seconds()
            data['last_price'] = float(d['LastPrice'])
            data['open_interest'] = float(d['OpenInterest'])
            data['ask_price1'] = float(d['AskPrice1'])
            data['ask_volume1'] = float(d['AskVolume1'])
            data['bid_price1'] = float(d['BidPrice1'])
            data['bid_volume1'] = float(d['BidVolume1'])
            data['turnover'] = float(d['Turnover'])
            data['volume'] = volume

            tmp_data.append(data)

    # Load the data of the last day.
    ret = _strip_features(tmp_data)
    if ret:
        states_list.append(ret[0])
        targets_list.append(ret[1])

    return states_list, targets_list


def _strip_features2(states,  # volatility, meanvalue,
                     target_steps, target_type,
                     minute_df, minite_15_df, day_df, week_df, month_df,
                     rise_th, still_th, make_sample=False):
    # State
    #  0: open
    #  1: close
    #  2: mid_price
    #  3: volume - the original "volume" is accumulated.
    #  4: open_interest_diff
    #  5: volume_direction
    #  6: volume_depth
    #  7: ups - tick number continuously goes up
    #  8: downs - tick number continuously goes down
    #  9: spread
    # 10: serial_id

    # states = pd.concat([states, meanvalue, volatility], axis=1)
    # print states.columns, states.shape
    states['open_interest_diff'] = states['interest'].diff()
    states['mid_price'] = (states['ask_price1'] + states['bid_price1']) / 2
    states['volume_direction'] = np.log(
        states['ask_volume1'] / states['bid_volume1'])
    states['volume_depth'] = states['ask_volume1'] + states['bid_volume1']
    states['ups'] = 0
    states['downs'] = 0
    states['serial_id'] = 0
    states['spread'] = states['ask_price1'] - states['bid_price1']
    states['spread2'] = states['ask_price1']*states['ask_volume1'] - \
        states['bid_price1']*states['bid_volume1']
    states['bid_all'] = states['bid_price1']*states['bid_volume1']
    states['ask_all'] = states['ask_price1']*states['ask_volume1']
    # states['mean_diff_interest'] = states['mean_interest'].diff()
    delta = (states.index - states.index[0]).total_seconds().tolist()
    # dateme = states.index[0]
    second_time = pd.Series(states.index)
    delta_time = pd.Timedelta(minutes=1)
    minute_time = second_time.apply(
        lambda t: (t - delta_time).replace(second=0, microsecond=0))

    delta_time = pd.Timedelta(minutes=15)
    minute_15_time = second_time.apply(
        lambda t: (t - delta_time - pd.Timedelta(minutes=t.minute % 15)).replace(second=0, microsecond=0))

    delta_time = pd.Timedelta(days=1)
    day_time = second_time.apply(lambda t: (t - delta_time).normalize())

    delta_time = pd.Timedelta(days=second_time.iloc[0].dayofweek+1)
    week_time = second_time.apply(lambda t: (t - delta_time).normalize())

    # delta_time = pd.Timedelta(days=second_time.iloc[0].day)
    # month_time = second_time.apply(lambda t: (t - delta_time).normalize())

    # for col in minute_df.columns:
    #     states['last_minute_' + col] = minute_df.ix[minute_time, col].tolist()

    # for col in minite_15_df.columns:
    #     states['last_15_minute_' + col] = \
    #         minite_15_df.ix[minute_15_time, col].tolist()

    # for col in day_df.columns:
    #     states['last_day_' + col] = day_df.ix[day_time, col].tolist()

    # for col in week_df.columns:
    #     states['last_week_' + col] = week_df.ix[week_time, col].tolist()

    # for col in month_df.columns:
    #     states['last_month_' + col] = month_df.ix[month_time, col].tolist()

    # Drop rows with NaN.
    if not make_sample:
        states.dropna(inplace=True)
    if states.empty:
        return (None, None, None)

    # Transform to numpy.
    states_np = states.as_matrix(columns=BASE_COLUMNS).astype(np.float32)
    klines_np = states.as_matrix(columns=KLINE_COLUMNS).astype(np.float32)
    # vola_np = volatility.as_matrix().astype(np.float64)
    length = len(states_np)
    max_target = max(target_steps)
    targets = np.empty((length - max_target - 1, len(target_steps)),
                       dtype=np.float32)
    # NOTE(reed): In softmax_cross_entropy, it takes -1 as ignore label.
    targets[:] = -1
    # print states_np[5, :]
    for j in range(1, length - max_target - 1):
        last_price = states_np[j-1, 1]
        # NOTE(lym): bid-ask-mean
        # this_price = (states_np[j, 19]+states_np[j, 21])/2.0
        this_price = states_np[j, 1]
        # NOTE(reed): Since states_np.dtype is float32, and the minimal
        # price change of IF is 0.2, we use 0.1 here to prevent from
        # floating error.
        if this_price > last_price + 0.003:
            states_np[j, 7] = states_np[j-1, 7] + 1
            states_np[j, 8] = 0
        elif this_price < last_price - 0.003:
            states_np[j, 7] = 0
            states_np[j, 8] = states_np[j-1, 8] + 1
        else:
            states_np[j, 7] = states_np[j-1, 7]
            states_np[j, 8] = states_np[j-1, 8]

        if target_type == 'c':  # classification
            for k, t in enumerate(target_steps):
                target_price = states_np[j+t, 1]
                ratio = target_price / this_price - 1
                if ratio > rise_th:
                    targets[j, k] = 0
                elif ratio < -rise_th:
                    targets[j, k] = 2
                elif -still_th < ratio < still_th:
                    targets[j, k] = 1

                # NOTE(reed): else the targets is left to be -1 as
                # ignored.
        elif target_type == 'c1':    # up up up down down down
            for k, t in enumerate(target_steps):
                # 19 --- bid_price1 21 --- ask_price1
                target_price = (states_np[j+1:j+t+1, 19] +
                                states_np[j+1:j+t+1, 21]) / 2.0
                # ratios = target_price.sum() / this_price - 1
                max1 = target_price.max()
                min1 = target_price.min()
                mean_ratio = target_price[-1] - this_price
                if mean_ratio > 0.003 and (this_price-min1) < (max1-this_price)*0.4:
                    targets[j, k] = 0
                if mean_ratio < -0.003 and (this_price-min1)*0.4 > (max1-this_price):
                    targets[j, k] = 1
                # print j, k, targets[j, k]
                if recordFlag:
                    # if targets[j, k] == 0 or targets[j, k] == 2:
                    recFile.write(str(states.index[j])+', '+str(j)+', ' +
                                  str(targets[j, k])+'\n')
                    recFile.flush()
        else:  # regression
            for k, t in enumerate(target_steps):
                # 19 --- bid_price1 21 --- ask_price1
                target_price = (states_np[j+1:j+t+1, 19] +
                                states_np[j+1:j+t+1, 21]) / 2.0
                # ratios = target_price.sum() / this_price - 1
                max1 = target_price.max()
                min1 = target_price.min()
                mean_ratio = target_price[-1] - this_price
                targets[j, k] = mean_ratio
                if recordFlag:
                    # if targets[j, k] == 0 or targets[j, k] == 2:
                    recFile.write(str(states.index[j])+', '+str(j)+', ' +
                                  str(targets[j, k])+'\n')
                    recFile.flush()
            # y = np.empty((max_target+1, ), dtype=np.float64)
            # A = np.ones((max_target+1, 2), dtype=np.float64)
            # for t in range(max_target+1):
            #     target_price = states_np[j+t, 1]
            #     y[t] = target_price / this_price
            #     A[t, 0] = delta[j+t]
            # # Use the first timestamp as original point.
            # A[:, 0] -= A[0, 0]

            # for k, t in enumerate(target_steps):
            #     m, _ = np.linalg.lstsq(A[:t+1], y[:t+1])[0]
            #     targets[j, k] = m

    if make_sample:
        targets = targets.astype(np.object)
        targets[:, 0] = [e for e in states.index[:targets.shape[0]]]
        targets = np.delete(targets, (0, ), axis=0)
        return (states_np[1: length - max_target - 1],
                None,
                targets)
    targets = np.delete(targets, (0, ), axis=0)
    # print targets

    return (states_np[1: length - max_target - 1],
            klines_np[1: length - max_target - 1],
            targets)


def load_gfund_data(contract, start_date, end_date, target_steps=[1, 2],
                    target_type='c', freq='1S', rise_th=1e-4, still_th=5e-5,
                    make_sample=False):
    logger.info("loading second k-lines")
    tmp_states_list = gfd.get_day_bar(
        contract, (start_date, end_date), freq=freq, merge=False,
        data_source=gfd.YINHE)

    tmp_states = pd.concat(tmp_states_list)

    d_dict = {'open': 'first',
              'high': 'max',
              'close': 'last',
              'low': 'min',
              'volume': 'sum'}

    minute_df = pd.DataFrame()
    minute_15_df = pd.DataFrame()
    day_df = pd.DataFrame()
    week_df = pd.DataFrame()
    month_df = pd.DataFrame()

    logger.info("generating minute k-lines")
    for col in ['open', 'high', 'low', 'close', 'volume']:
        minute_df[col] = tmp_states[col].resample('T').apply(d_dict[col])
    minute_df.dropna(inplace=True)

    logger.info("generating 15 minute k-lines")
    for col in ['open', 'high', 'low', 'close', 'volume']:
        minute_15_df[col] = tmp_states[col].resample('15T').apply(d_dict[col])
    minute_15_df.dropna(inplace=True)

    logger.info("generating day k-lines")
    for col in minute_df.columns:
        day_df[col] = minute_df[col].resample('D').apply(d_dict[col])
    # NOTE(reed): Use pad method to fillna, so we could take last calendar data
    # as last trading day.
    day_df.fillna(method='pad', inplace=True)

    # NOTE(reed): Since the number of trading days in each week/month is
    # various, we use 'mean' instead of 'sum'.
    d_dict['volume'] = 'mean'
    logger.info("generating week k-lines")
    for col in day_df.columns:
        week_df[col] = day_df[col].resample('W').apply(d_dict[col])

    logger.info("generating month k-lines")
    for col in day_df.columns:
        month_df[col] = day_df[col].resample('M').apply(d_dict[col])

    # volatility = cp.load(open('../privateDataAndTools/volatility.pkl', 'r'))
    # meanvalue = cp.load(open('../privateDataAndTools/mean.pkl', 'r'))
    # print("len of states %s, len of volatility %s, len of mean %s" %
    #       (len(tmp_states_list), len(volatility), len(meanvalue)))
    pool = mp.Pool(processes=10)
    states_list = []
    klines_list = []
    targets_list = []
    res_list = []
    for i in range(len(tmp_states_list)):
        res_list.append(pool.apply_async(
            _strip_features2, (tmp_states_list[i],  # volatility[i], meanvalue[i],
                               target_steps, target_type,
                               minute_df, minute_15_df, day_df, week_df,
                               month_df,
                               rise_th, still_th,
                               make_sample)))
    pool.close()
    pool.join()

    res_list = filter(lambda (x, y, z): x is not None,
                      map(lambda ret: ret.get(), res_list))
    for states, klines, targets in res_list:
        states_list.append(states)
        klines_list.append(klines)
        targets_list.append(targets)
    # print targets_list

    return states_list, klines_list, targets_list


def extract_indicator(df):
    steps = 60
    m1 = Indicator.KDJ()
    m1.filldata(df)
    m2 = Indicator.MACD()
    m2.filldata(df)
    m3 = Indicator.BOLL()
    m3.filldata(df)
    m4 = Indicator.RSI()
    m4.filldata(df)
    m5 = Indicator.OBV()
    m5.filldata(df)
    m6 = Indicator.MA()
    m6.filldata(df)

    alldata = pd.concat([m1.kdj, m2.macd, m3.boll, m4.rsi, m5.obv, m6.ma], axis=1)
    # print alldata.columns, alldata.shape
    # remove keypoint can't spread steps
    # and split pos samples and neg samples
    whole = m1.get_keypoint()
    print whole.index[0], len(whole)
    valid_whole = m1.data.index.get_indexer(whole.index)
    valid_whole = valid_whole[valid_whole > steps]
    valid_whole = m1.data.index[valid_whole]

    whole = whole.loc[valid_whole]
    m1.keypoint = whole
    m1.verify_keypoint(0.3)
    pos = m1.keypoint.index

    whole.loc[pos] = None
    whole.dropna(inplace=True)
    neg = whole.index

    pos_index = alldata.index.get_indexer(pos)
    neg_index = alldata.index.get_indexer(neg)
    pos_samples = []
    neg_samples = []
    for i in pos_index:
        element = alldata.iloc[i-steps:i]
        pos_samples.append(element.as_matrix())
    # print [e.shape for e in pos_samples]
    pos_samples = np.stack(pos_samples)
    pos_flags = np.ones((pos_samples.shape[0], 1))
    for i in neg_index:
        element = alldata.iloc[i-steps:i]
        neg_samples.append(element.as_matrix())
    neg_samples = np.stack(neg_samples)
    neg_flags = np.zeros((neg_samples.shape[0], 1))
    samples = np.concatenate([pos_samples, neg_samples])
    flags = np.concatenate([pos_flags, neg_flags])
    return samples, [], flags


def load_indicator_data(contract, start_date, end_date, target_steps=[1, 2],
                        target_type='c', freq='20S', rise_th=1e-4, still_th=5e-5,
                        make_sample=False):
    logger.info("loading second k-lines")
    tmp_states_list = gfd.get_day_bar(
        contract, (start_date, end_date), freq=freq, merge=False,
        data_source=gfd.YINHE)

    pool = mp.Pool(processes=10)
    states_list = []
    klines_list = []
    targets_list = []
    res_list = []
    for i in range(len(tmp_states_list)):
        res_list.append(pool.apply_async(
            extract_indicator, (tmp_states_list[i], )))
    pool.close()
    pool.join()

    res_list = filter(lambda (x, y, z): x is not None,
                      map(lambda ret: ret.get(), res_list))
    for states, klines, targets in res_list:
        states_list.append(states)
        klines_list.append(klines)
        targets_list.append(targets)

    return states_list, klines_list, targets_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--windowlen', '-w', type=int, default=30,
                        help='Number of lines for each data widow slice')
    parser.add_argument("--data_start_date", type=int, default=20150101,
                        help="data start date")
    parser.add_argument("--data_end_date", type=int, default=20161231,
                        help="data end date")
    parser.add_argument("--log_file", help="Log file name.")
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO", help="Log level.")
    parser.add_argument(
        "--target_type", choices=["c", "r"], default="c",
        help="target type: c for classification, r for regression.")

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

    logger.info("Loading data...")
    # target steps in the next 1, 2, ..., 10 steps.
    target_steps = [5]
    # states_list, targets_list = load_history_data(
    #      args.data_start_date, args.data_end_date, target_steps=target_steps,
    #      target_type=args.target_type)

    states_list, klines_list, targets_list = load_gfund_data(
         'T', args.data_start_date, args.data_end_date,
         target_steps=target_steps, target_type=args.target_type)

    days = len(states_list)
    logger.info("Load %d days data.", days)

    filename = 'dataset2_%s_%d_%d_T_step5.h5' % (
        args.target_type, args.data_start_date, args.data_end_date)
    file = h5py.File(filename, 'w')
    file.create_dataset('days', data=days)
    for i in range(days):
        file.create_dataset('states_%d' % i, data=states_list[i])
        file.create_dataset('klines_%d' % i, data=klines_list[i])
        file.create_dataset('targets_%d' % i, data=targets_list[i])
    file.close()
