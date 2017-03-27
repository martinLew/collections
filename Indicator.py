# coding: utf-8
import pandas as pd
import gfunddataprocessor.gfunddataprocessor as gfd
import numpy as np
import sys

from exceptions import NotImplementedError, AttributeError


class Indicator(object):

    def __init__(self):
        self.data = self.data_np = None
        self.keypoint = None
        pass

    def filldata(self, data):
        raise NotImplementedError

    def get_keypoint(self):
        raise NotImplementedError

    def verify_keypoint(self):
        raise NotImplementedError

    def drawlines(self, path):
        raise NotImplementedError


class MACD(Indicator):
    """
        pass
    """
    def __init__(self, slow=26, fast=12, aver=9):
        super(MACD, self).__init__()
        self.slow = slow
        self.fast = fast
        self.aver = aver
        self.macd = None
        self.keypoint = None

    def filldata(self, data):
        """
            receive data(pandas.DataFrame) and calculate
            the MACD indicator
        """
        if self.macd is not None:
            return
        if 'close' not in data.columns:
            raise ValueError("lastprice not found in data")
        self.data = data
        self.data_np = data_np = data.as_matrix(columns=['close', ]).astype(np.float64)
        macd = np.empty((len(data), 3), dtype=np.float64)
        prev_ema_slow = prev_ema_fast = data_np[0, 0]
        for i in range(len(data)):
            if i == 0:
                macd[0, :] = 0
                continue
            # calculate ema26, 12(classical), dif
            ema_slow = self.calculateema(self.slow, data_np[i, 0], prev_ema_slow)
            ema_fast = self.calculateema(self.fast, data_np[i, 0], prev_ema_fast)
            macd[i, 0] = ema_fast - ema_slow
            prev_ema_fast = ema_fast
            prev_ema_slow = ema_slow

        # calculate dea, macd
        for i in range(1, len(macd)):
            macd[i, 1] = self.calculateema(self.aver, macd[i, 0], macd[i-1, 1])
            macd[i, 2] = (macd[i, 0] - macd[i, 1])*2
        self.macd = pd.DataFrame(macd, index=data.index,
                                 columns=['dif', 'dea', 'macd'])

    def get_keypoint(self):
        if self.macd is None:
            raise IndexError("macd not created yet!")
        if self.keypoint is not None:
            return self.keypoint
        self.keypoint = pd.DataFrame(index=self.macd.index, columns=['type', ])
        macd = self.macd.as_matrix()
        for i in range(2, len(self.macd)):
            if (macd[i, 2] > 0 and macd[i-1, 2] < 0):
                self.keypoint.iloc[i, 0] = 1
            elif (macd[i, 2] < 0 and macd[i-1, 2] > 0):
                self.keypoint.iloc[i, 0] = -1
        self.keypoint.dropna(inplace=True)
        return self.keypoint

    def calculateema(self, period, now, prev):
        return now*2.0/(period+1) + prev*float(period-1)/(period+1)

    def verify_keypoint(self, thresh):
        correct = 0
        for i in range(1, len(self.keypoint.index)):
            begin = self.keypoint.index[i-1]
            end = self.keypoint.index[i]
            ser = pd.date_range(begin, end, freq='1S')
            ldata = self.data.loc[ser, 'close']
            try:
                difdata = ldata - ldata.iloc[0]
            except:
                print len(ldata)
            if (self.keypoint.iloc[i-1, 0] == 1 and
                (difdata > thresh).sum() >= 1) \
                    or (self.keypoint.iloc[i-1, 0] == -1
                        and (difdata < -thresh).sum() >= 1):
                correct += 1
        return correct
        # print('correct trigger num: %s' % correct)

    def drawlines(self, path):
        if self.data is None or self.keypoint is None:
            raise AttributeError("data or keypoint not initilized")
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        xline1 = [self.macd.index.indexer_at_time(e)[0]
                  for e in self.keypoint.index]
        macd = self.macd.as_matrix()
        plt.figure(1, figsize=(200, 20))
        xline = range(len(self.macd))
        plt.subplot(2, 1, 1)
        yline = self.data_np
        plt.plot(xline, yline, 'r-')
        plt.plot(xline1, yline[xline1], '*')
        plt.subplot(2, 1, 2)
        yline1 = macd[:, 0]
        yline2 = macd[:, 1]
        plt.plot(xline, yline2, 'y-')
        plt.plot(xline, yline1, 'r-')
        plt.plot(xline1, yline1[xline1], '*')
        plt.savefig(path)


class KDJ(Indicator):
    def __init__(self, cycle=9, sm1=3, sm2=3):
        super(KDJ, self).__init__()
        self.cycle = cycle
        self.sm1 = sm1
        self.sm2 = sm2
        self.kdj = None

    def filldata(self, data):
        """
            receive data(pandas.DataFrame) and calculate
            the KDJ indicator
        """
        if self.kdj is not None:
            return
        if 'close' not in data.columns:
            raise ValueError("lastprice not found in data")
        self.data = data
        self.data_np = data_np = data.as_matrix(columns=['close', ]) \
            .astype(np.float64)
        kdj = np.empty((len(data), 3), dtype=np.float64)
        RSV = self.data_np.copy()
        for i in range(self.cycle-1, len(self.data_np)):
            cycle_max = data_np[i-self.cycle+1:i+1].max()
            cycle_min = data_np[i-self.cycle+1:i+1].min()
            if cycle_max == cycle_min:
                RSV[i] = 100.0
            else:
                RSV[i] = (data_np[i]-cycle_min) / (cycle_max-cycle_min) * 100.0
        RSV[0:self.cycle-1] = RSV[self.cycle-1]

        # calculate K, D
        for i in range(len(data)):
            if i == 0:
                kdj[i, 0] = RSV[0]
                continue
            kdj[i, 0] = (self.sm1-1.0)/self.sm1*kdj[i-1, 0] + \
                1.0/self.sm1*RSV[i]
        for i in range(len(data)):
            if i == 0:
                kdj[i, 1] = kdj[i, 0]
                continue
            kdj[i, 1] = (self.sm2-1.0)/self.sm2*kdj[i-1, 1] + \
                1.0/self.sm2*kdj[i, 0]
        kdj[:, 2] = 3.0*kdj[:, 1] - 2.0*kdj[:, 0]
        self.kdj = pd.DataFrame(kdj, columns=['K', 'D', 'J'],
                                index=data.index)

    def get_keypoint(self):
        if self.kdj is None:
            raise IndexError("kdj not created yet!")
        if self.keypoint is not None:
            return self.keypoint
        self.keypoint = pd.DataFrame(index=self.kdj.index, columns=['type', ])
        kdj = self.kdj.as_matrix()
        diff = kdj[:, 0] - kdj[:, 1]
        for i in range(2, len(diff)):
            if (diff[i] > 0 and diff[i-1] < 0):
                self.keypoint.iloc[i, 0] = 1
            elif (diff[i] < 0 and diff[i-1] > 0):
                self.keypoint.iloc[i, 0] = -1
        self.keypoint.dropna(inplace=True)
        return self.keypoint

    def verify_keypoint(self, thresh):
        correct = 0
        real = []
        for i in range(1, len(self.keypoint.index)):
            begin = self.keypoint.index[i-1]
            end = self.keypoint.index[i]
            ser = pd.date_range(begin, end, freq='20S')
            ldata = self.data.loc[ser, 'close']
            try:
                difdata = ldata.diff()
            except:
                print len(ldata)
            if (self.keypoint.iloc[i-1, 0] == 1 and
                (difdata > 1e-4).sum()/float(len(difdata)) >= thresh) \
                    or (self.keypoint.iloc[i-1, 0] == -1
                        and (difdata < -1e-4).sum()/float(len(difdata)) >= thresh):
                real.append(begin)
                correct += 1
        self.keypoint = self.keypoint.loc[real, :]
        return correct
        # print('correct trigger num: %s' % correct)

    def drawlines(self, path):
        if self.data is None or self.keypoint is None:
            raise AttributeError("data or keypoint not initilized")
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        xline1 = [self.kdj.index.indexer_at_time(e)[0]
                  for e in self.keypoint.index]
        kdj = self.kdj.as_matrix()
        plt.figure(1, figsize=(200, 20))
        xline = range(len(self.kdj))
        plt.subplot(2, 1, 1)
        yline = self.data_np
        plt.plot(xline, yline, 'r-')
        plt.plot(xline1, yline[xline1], '*')
        plt.subplot(2, 1, 2)
        yline1 = kdj[:, 0]
        yline2 = kdj[:, 1]
        plt.plot(xline, yline2, 'y-')
        plt.plot(xline, yline1, 'r-')
        plt.plot(xline1, yline1[xline1], '*')
        plt.savefig(path)


class RSI(Indicator):
    def __init__(self, slow=12, fast=6):
        super(RSI, self).__init__()
        self.slow = slow
        self.fast = fast
        self.rsi = None

    def filldata(self, data):
        """
            receive data(pandas.DataFrame) and calculate
            the KDJ indicator
        """
        if self.rsi is not None:
            return
        if 'close' not in data.columns:
            raise ValueError("lastprice not found in data")
        self.data = data
        self.data_np = data_np = data.as_matrix(columns=['close']) \
            .astype(np.float64)
        rsi = np.empty((len(data), 2), dtype=np.float64)
        diff = data_np.copy()
        diff[1:] = data_np[1:] - data_np[:-1]
        for i in range(self.fast, len(self.data_np)):
            target = diff[i-self.fast+1:i+1]
            pos = target[target > 0].sum()
            neg = -1.0 * target[target < 0].sum()
            if pos+neg == 0:
                rsi[i, 0] = 50.0
            else:
                rsi[i, 0] = pos / (pos+neg) * 100.0
        rsi[0:self.fast, 0] = rsi[self.fast, 0]
        for i in range(self.slow, len(self.data_np)):
            target = diff[i-self.slow+1:i+1]
            pos = target[target > 0].sum()
            neg = -1.0 * target[target < 0].sum()
            if pos+neg == 0:
                rsi[i, 1] = 50.0
            else:
                rsi[i, 1] = pos / (pos+neg) * 100.0
        rsi[0:self.slow, 1] = rsi[self.slow, 1]
        self.rsi = pd.DataFrame(rsi, index=data.index,
                                columns=['fast', 'slow'])

    def get_keypoint(self):
        if self.rsi is None:
            raise IndexError("rsi not created yet!")
        if self.keypoint is not None:
            return self.keypoint
        self.keypoint = pd.DataFrame(index=self.rsi.index, columns=['type', ])
        rsi = self.rsi.as_matrix()
        diff = rsi[:, 0] - rsi[:, 1]
        for i in range(2, len(diff)):
            if (diff[i] > 0 and diff[i-1] < 0):
                self.keypoint.iloc[i, 0] = 1
            elif (diff[i] < 0 and diff[i-1] > 0):
                self.keypoint.iloc[i, 0] = -1
        self.keypoint.dropna(inplace=True)
        return self.keypoint

    def verify_keypoint(self, thresh):
        correct = 0
        for i in range(1, len(self.keypoint.index)):
            begin = self.keypoint.index[i-1]
            end = self.keypoint.index[i]
            ser = pd.date_range(begin, end, freq='1S')
            ldata = self.data.loc[ser, 'close']
            try:
                difdata = ldata - ldata.iloc[0]
            except:
                print len(ldata)
            if (self.keypoint.iloc[i-1, 0] == 1 and
                (difdata > thresh).sum() >= 1) \
                    or (self.keypoint.iloc[i-1, 0] == -1
                        and (difdata < -thresh).sum() >= 1):
                correct += 1
        print('correct trigger num: %s' % correct)

    def drawlines(self, path):
        if self.data is None or self.keypoint is None:
            raise AttributeError("data or keypoint not initilized")
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        xline1 = [self.rsi.index.indexer_at_time(e)[0]
                  for e in self.keypoint.index]
        rsi = self.rsi.as_matrix()
        plt.figure(1, figsize=(200, 20))
        xline = range(len(self.rsi))
        plt.subplot(2, 1, 1)
        yline = self.data_np
        plt.plot(xline, yline, 'r-')
        plt.plot(xline1, yline[xline1], '*')
        plt.subplot(2, 1, 2)
        yline1 = rsi[:, 0]
        yline2 = rsi[:, 1]
        plt.plot(xline, yline2, 'y-')
        plt.plot(xline, yline1, 'r-')
        plt.plot(xline1, yline1[xline1], '*')
        plt.savefig(path)


class OBV(Indicator):
    def __init__(self):
        super(OBV, self).__init__()
        self.obv = None

    def filldata(self, data):
        """
            receive data(pandas.DataFrame) and calculate
            the KDJ indicator
        """
        if self.obv is not None:
            return
        if 'close' not in data.columns:
            raise ValueError("lastprice not found in data")
        self.data = data
        self.data_np = data_np = data.as_matrix(columns=['close', 'volume']) \
            .astype(np.float64)
        obv = np.empty((len(data), 1), dtype=np.float64)
        for i in range(len(self.data_np)):
            if i == 0:
                obv[i] = 0
                continue
            sign = 0
            if data_np[i, 0] > data_np[i-1, 0]:
                sign = 1
            elif data_np[i, 0] < data_np[i-1, 0]:
                sign = -1
            obv[i] = obv[i-1]+sign*data_np[i, 1]
        self.obv = pd.DataFrame(obv, index=data.index, columns=['value'])

    def drawlines(self, path):
        if self.data is None:
            raise AttributeError("data or keypoint not initilized")
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        obv = self.obv.as_matrix()
        plt.figure(1, figsize=(200, 20))
        xline = range(len(self.obv))
        plt.subplot(2, 1, 1)
        yline = self.data_np[:, 0]
        plt.plot(xline, yline, 'r-')
        plt.subplot(2, 1, 2)
        yline1 = obv[:, 0]
        plt.plot(xline, yline1, 'r-')
        plt.savefig(path)


class BOLL(Indicator):
    def __init__(self, period=26, times=2):
        super(BOLL, self).__init__()
        self.period = period
        self.times = times
        self.boll = None

    def filldata(self, data):
        """
            receive data(pandas.DataFrame) and calculate
            the KDJ indicator
        """
        if self.boll is not None:
            return
        if 'close' not in data.columns:
            raise ValueError("lastprice not found in data")
        self.data = data
        self.data_np = data_np = data.as_matrix(columns=['close']) \
            .astype(np.float64)
        boll = np.empty((len(data), 3), dtype=np.float64)
        for i in range(self.period-1, len(self.data_np)):
            tmpdata = data_np[i-self.period+1:i+1]
            aver = tmpdata.mean()
            std = np.sqrt(((tmpdata-aver)*(tmpdata-aver)).sum()/self.period)
            boll[i, 0] = aver
            boll[i, 1] = aver + self.times*std
            boll[i, 2] = aver - self.times*std
        boll[:self.period-1, :] = boll[self.period-1, :]
        self.boll = pd.DataFrame(boll, index=data.index,
                                 columns=['MID', 'UP', 'DOWN'])

    def drawlines(self, path):
        if self.data is None:
            raise AttributeError("data or keypoint not initilized")
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        boll = self.boll.as_matrix()
        plt.figure(1, figsize=(200, 20))
        xline = range(len(self.boll))
        plt.subplot(2, 1, 1)
        yline = self.data_np[:, 0]
        plt.plot(xline, yline, 'r-')
        plt.subplot(2, 1, 2)
        yline1 = boll[:, 0]
        yline2 = boll[:, 1]
        yline3 = boll[:, 2]
        plt.plot(xline, yline1, 'r-')
        plt.plot(xline, yline2, 'g-')
        plt.plot(xline, yline3, 'b-')
        plt.savefig(path)


class MA(Indicator):
    def __init__(self, period=10):
        super(MA, self).__init__()
        self.period = period
        self.ma = None

    def filldata(self, data):
        """
            receive data(pandas.DataFrame) and calculate
            the KDJ indicator
        """
        if self.ma is not None:
            return
        if 'close' not in data.columns:
            raise ValueError("lastprice not found in data")
        self.data = data
        self.data_np = data_np = data.as_matrix(columns=['close']) \
            .astype(np.float64)
        ma = np.empty((len(data), 1), dtype=np.float64)
        for i in range(self.period-1, len(self.data_np)):
            tmpdata = data_np[i-self.period+1:i+1]
            aver = tmpdata.mean()
            ma[i, 0] = aver
        ma[:self.period-1, :] = ma[self.period-1, :]
        self.ma = pd.DataFrame(ma, index=data.index,
                               columns=['MA'])

    def drawlines(self, path):
        if self.data is None:
            raise AttributeError("data or keypoint not initilized")
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        ma = self.ma.as_matrix()
        plt.figure(1, figsize=(200, 20))
        xline = range(len(self.ma))
        yline = self.data_np[:, 0]
        plt.plot(xline, yline, 'r-')
        yline1 = ma[:, 0]
        plt.plot(xline, yline1, 'g-')
        plt.savefig(path)

if __name__ == "__main__":
    df_list = gfd.get_day_bar('T', (sys.argv[1], sys.argv[1]), freq='20S',
                              data_source=gfd.YINHE, merge=False)
    # m1 = MACD(26, 12, 9)
    m1 = MA(10)
    m2 = KDJ(9, 3, 3)
    # m1 = RSI(60, 30)
    # m1 = OBV()
    for df in df_list:
        m2.kdj = None
        m2.keypoint = None
        m1.filldata(df)
        m2.filldata(df)
        points = m2.get_keypoint()
        indexes = points.index
        sel = []
        # for index in indexes:
        #     if points.loc[index][0] == -1 and df.loc[index, 'bid_price1'] > m1.ma.loc[index][0]:
        #         sel.append(index)
        #     if points.loc[index][0] == 1 and df.loc[index, 'ask_price1'] < m1.ma.loc[index][0]:
        #         sel.append(index)
        # m2.keypoint = points.loc[sel, :]
        # points = m1.get_keypoint()
        cn = m2.verify_keypoint(0.005*3.5)
        print("%s, %s" % (len(m2.keypoint), cn))
        m2.drawlines('./sample.png')
    # m1 = BOLL(60, 2)
    # m1.filldata(d1)
    # points = m1.get_keypoint()
    # print(len(points))
    # m1.verify_keypoint(0.005*3)
    # m1.drawlines('./sample.png')
