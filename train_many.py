import datetime
import sys
import time

import numpy as np
from sklearn import linear_model
from sklearn.externals import joblib

from StockLoader import StockLoader
from TweetLoader import TweetLoader

SAVE_DIR_PATH = './pickles/result_1.2/'
STOCK_DIR_PATH = '/Users/Opi/dev/data/stockData/data_1.2'
message = 'processing..'

def train_clfs(clfs, X, date, stock, unique_class):
    message = 'for training... it takes:'
    for i, clf in enumerate(clfs):
        start_time = time.time()
        y = stock.get_batch_stock(i, date, len(X))
        clf.partial_fit(X,y,unique_class)
        print(stock.stock[i][0], message, time.time() - start_time)

def pkl_clfs(clfs, stock_batch, start, end, minute):
    for i, clf in enumerate(clfs):
        pkl_name = '{0}_{1}_{2}_{3}_timeshift:{4}.pkl'.format(type(clf).__name__,
                                                          stock_batch[i][0],
                                                          start,
                                                          end,
                                                          minute)
        joblib.dump(clf, SAVE_DIR_PATH + pkl_name)

if __name__ == '__main__':
    nth = int(sys.argv[1])

    stock = StockLoader(STOCK_DIR_PATH, batch_size=20)
    stock.set_nth_batch(nth)

    tweet = TweetLoader()

    clfs = []
    for _ in range(len(stock.stock_batch)):
        clfs.append(linear_model.SGDClassifier(n_jobs=-1))

    unique_class = np.array([-1,1])
    start = '2017-04-17'
    end = '2017-05-30'
    minute = 0
    timeshift = datetime.timedelta(minutes=minute)

    partial_fit_threshold = 100000 # X를 모았다가 임계 이상이 되면 partial_fit을 호출한다.
    X = []
    y_source = []

    for partial_X, date in tweet.gen_X(start, end, shift=timeshift): # X를 구한다.
        print(date, 'processing....')
        X.extend(partial_X)
        y_source.append((date, len(partial_X)))
        if len(X) > partial_fit_threshold: # X를 충분히 많이 모았으면 partial_fit을 호출한다.
            print('partial_fit calling phase.. y_source len :', len(y_source))
            for index, clf in enumerate(clfs):
                y = stock.get_y(index, y_source)
                start_time = time.time()
                clf.partial_fit(X, y, unique_class)
                print(stock.stock_batch[index][0], message, time.time() - start_time)
            X.clear()
            y_source.clear()


            #train_clfs(clfs, X, date, stock, unique_class)
        #y = stock.get_stock_arr(date, len(X))
        #clf.partial_fit(X, y, classes=unique_class)

    pkl_clfs(clfs, stock.stock_batch, start, end, minute)

    print('hellp scikit')
    """
    1. 주식정보를 읽는다
    2. classifier를 주식 수 만큼 생성한다
    3. """





