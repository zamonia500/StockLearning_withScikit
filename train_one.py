import datetime
import sys
import time

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib

from StockLoader import StockLoader
from TweetLoader import TweetLoader

SAVE_DIR_PATH = './pickles/result_1.2/'
STOCK_DIR_PATH = '/Users/Opi/dev/data/stockData/data_1.2'
message = 'processing..'

START = '2017-04-17'
END = '2017-05-30'

def get_y(stock, length, date):
    for _ in range(15):  # 만약 원하는 날짜가 주말또는 공휴일이라 값이 존재하지 않으면 다음날을 찾아본다.
        if date in stock:
            value = stock[date]
            break
        else:
            vars = list(map(int, date.split('-')))
            next_date = datetime.datetime(vars[0], vars[1], vars[2]) + datetime.timedelta(days=1)
            date = '{0}-{1:02d}-{2:02d}'.format(next_date.year, next_date.month, next_date.day)
    y = np.ndarray(shape=(length,), dtype=int)
    y.fill(value)

    return y

if __name__ == '__main__':
    # 회사 이름을 하나 받아서 start ~ end 까지 학습시킨 후 저장한다.
    stocks = StockLoader(STOCK_DIR_PATH)
    stock_name = 'LG전자.txt'
    stock = stocks.get_one_stock(stock_name)

    tweets = TweetLoader()
    minute = 0
    timeshift = datetime.timedelta(minutes=minute)

    clf = SGDClassifier(n_jobs=-1)
    classes = [-1,1]

    for X, date in tweets.gen_X(START, END, timeshift):
        y = get_y(stock, len(X), date)
        start_time = time.time()
        clf.partial_fit(X, y, classes=classes)
        print(stock_name, date, time.time() - start_time)

    pkl_name = '{0}_{1}_{2}_{3}_timeshift:{4}.pkl'.format(type(clf).__name__,
                                                          stock_name,
                                                          START,
                                                          END,
                                                          minute)
    joblib.dump(clf, SAVE_DIR_PATH + pkl_name)