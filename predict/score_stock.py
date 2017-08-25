import os
import glob
import sys
import datetime
import numpy as np

from sklearn import linear_model
from sklearn.externals import joblib
from TweetLoader import TweetLoader
from StockLoader import StockLoader

def get_pickles(dir_path):
    if not os.path.isdir(dir_path):
        raise ValueError('dir_path not exist')

    pickle_files = glob.glob(os.path.join(dir_path, '*.pkl'))

    pickle_list = []

    # example of pkl file name : SGDClassifier_AK홀딩스.txt_2017-04-17_2017-05-30_timeshift/0.pkl
    for pickle_file in pickle_files:
        pickle_object = joblib.load(pickle_file)
        pickle_info = os.path.basename(pickle_file).split('_')
        pickle_list.append({
            'scikit_object':pickle_object,
            'name':pickle_info[1],
            'start':pickle_info[2],
            'end':pickle_info[3],
            'shift':pickle_info[4].split('.')[0][10:], # ex) timeshift/0.pkl -> 0
            'score':[] # the result of score will appended here date by date
        })

    return pickle_list

def make_y(length, value):
    y = np.ndarray(shape=(length, 1), dtype=int)
    y.fill(value)
    return y


if __name__ == '__main__':
    """usage : python sys.argv[0] pickle_dir_path"""
    pickle_dir_path = sys.argv[1]
    pickle_list = get_pickles(pickle_dir_path)

    tweets = TweetLoader()
    timeshift = datetime.timedelta(minutes=int(pickle_list[0]['shift']))

    stocks = StockLoader('/Users/Opi/dev/data/stockData/data_1.2')
    stock_dic = stocks.get_dic_for_predict()

    # make_y = lambda len_x, y: np.ndarray(shape=(len_x,), dtype=float).fill(y)

    for X, _date in tweets.gen_X('2017-05-31', '2017-07-01', timeshift):
        for pkl in pickle_list:
            y = make_y(len(X), stock_dic[pkl['name']][_date])
            #predict_y = pkl['scikit_object'].predict(X)
            score = pkl['scikit_object'].score(X, y)
            print(pkl['name'], _date, len(X), score)
        #모든 피클들에 대해서 score
        #score 결과 update