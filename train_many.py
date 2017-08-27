import datetime
import os
import time
import argparse

import numpy as np
from sklearn import linear_model
from sklearn.externals import joblib

from StockLoader import StockLoader
from TweetLoader import TweetLoader


# StockLoader, TweetLoader에 넘겨주는 모든 path parameter는 train_*.py에서 선언 후 넘겨준다.
SAVE_DIR_PATH = './pickles/result_1.1'
STOCK_DIR_PATH = '/Users/Opi/dev/data/stockData/data_1.1'
VECTOR_DIR_PATH = '/Users/Opi/dev/data/tweet_index_vector/vectors'

message = 'processing ..'

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
        joblib.dump(clf, os.path.join(SAVE_DIR_PATH, pkl_name))

if __name__ == '__main__':
    """
    batch_size & nth_batch는 입력을 통해서 받는 것으로 한다
    usage : python argv[0] batch_size=50 nth_batch=0
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', action="store", dest="batch_size", type=int) # argument for use how many stock data for learning
    parser.add_argument('-nth_batch', action="store", dest="nth_batch", type=int)
    parser.add_argument('-start', action="store", dest="start", default='2017-04-17') # argument for the duration of tweet infomation to use for learning
    parser.add_argument('-end', action="store", dest="end", default='2017-05-31')
    parser.add_argument('-shift_time', action="store", dest="shift_time", type=int, default=0) # argument for shift_time
    parser.add_argument('-threshold', action="store", dest="enter_partial_fit_threshold", type=int, default=200000)
    args = parser.parse_args()
    del parser

    SAVE_DIR_PATH = os.path.join(SAVE_DIR_PATH, 'shift=' + str(args.shift_time))
    if not os.path.isdir(SAVE_DIR_PATH):
        os.mkdir(SAVE_DIR_PATH)

    stock = StockLoader(STOCK_DIR_PATH, batch_size=args.batch_size, nth_batch=args.nth_batch)
    tweet = TweetLoader(VECTOR_DIR_PATH)

    clfs = []
    for _ in range(len(stock.stock_batch)):
        clfs.append(linear_model.SGDClassifier(n_jobs=-1))

    classes = np.array([0,1])

    X = []
    y_source = []

    for partial_X, date in tweet.gen_X(args.start, args.end,
                                       shift=datetime.timedelta(minutes=args.shift_time)): # X를 구한다.
        print(date, message)
        X.extend(partial_X)
        y_source.append((date, len(partial_X)))
        if len(X) > args.enter_partial_fit_threshold: # X를 충분히 많이 모았으면 partial_fit을 호출한다.
            print('partial_fit calling phase.. y_source len :', len(y_source))
            for index, clf in enumerate(clfs):
                y = stock.get_y(index, y_source)
                start_time = time.time()
                clf.partial_fit(X, y, classes)
                print(stock.stock_batch[index][0], message, time.time() - start_time)
            X.clear()
            y_source.clear()

    pkl_clfs(clfs, stock.stock_batch, args.start, args.end, args.minute)
    print('end learning')






