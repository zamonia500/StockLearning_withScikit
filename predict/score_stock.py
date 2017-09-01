import os
import glob
import sys
import datetime
import argparse
import numpy as np

from sklearn import linear_model
from sklearn.externals import joblib
from TweetLoader import TweetLoader
from StockLoader import StockLoader

SAVE_DIR_PATH = './pickles/result_1.1'
STOCK_DIR_PATH = '/Users/Opi/dev/data/stockData/data_1.1'
VECTOR_DIR_PATH = '/Users/Opi/dev/data/tweet_index_vector/vectors'

"""
1. result_1.1 폴더안에 있는 모든 폴더 이름을 가져온다.
2. 각 폴더에 대하여 다음을 반복한다.
3. 폴더 안의 모든 피클에 대하여 반복한다.
4. 피클을 로드한다.
5. 피클의 이름을 가지고 name, start, end, timeshift를 추출한다.
6. 나머지 영역에 대하여 score를 한다.
7. 결과를 저장한다.?"""
def get_pickles(dir_path):
    if not os.path.isdir(dir_path):
        raise ValueError('dir_path not exist')

    pickle_files = glob.glob(os.path.join(dir_path, '*.pkl'))

    pickle_list = []

    # _example of pkl file name : SGDClassifier_AK홀딩스.txt_2017-04-17_2017-05-30_timeshift/0.pkl
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-start', action="store", dest="start", default='2017-06-04')  # argument for the duration of tweet infomation to use for learning
    parser.add_argument('-end', action="store", dest="end", default='2017-07-01')
    parser.add_argument('-threshold', action="store", dest="enter_partial_fit_threshold", type=int, default=10000)
    parser.add_argument('-pkl_dir_path', action="store", dest="pkl_dir_path", type=str)
    args = parser.parse_args()
    del parser

    pickle_list = get_pickles(args.pkl_dir_path)

    tweets = TweetLoader(VECTOR_DIR_PATH)
    timeshift = datetime.timedelta(minutes=int(pickle_list[0]['shift']))

    stocks = StockLoader(STOCK_DIR_PATH)
    stock_dic = stocks.get_dic_for_predict()

    for X, _date in tweets.gen_X(args.start, args.end, timeshift):
        for _ in range(15):  # 만약 원하는 날짜가 주말또는 공휴일이라 값이 존재하지 않으면 다음날을 찾아본다.
            if _date in stock_dic[pickle_list[0]['name']]:
                break
            else:
                vars = list(map(int, _date.split('-')))
                next_date = datetime.datetime(vars[0], vars[1], vars[2]) + datetime.timedelta(days=1)
                _date = '{0}-{1:02d}-{2:02d}'.format(next_date.year, next_date.month, next_date.day)
        for pkl in pickle_list:
            y = make_y(len(X), stock_dic[pkl['name']][_date])
            #predict_y = pkl['scikit_object'].predict(X)
            score = pkl['scikit_object'].score(X, y)
            print(pkl['name'], _date, len(X), score)
            break

        #모든 피클들에 대해서 score
        #score 결과 update