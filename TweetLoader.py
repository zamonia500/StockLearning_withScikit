import os
import ast
import numpy as np
import collections
import datetime
"""
클래스를 만들 것이다.
생성자로 start, end를 입력하여 언제부터 언제까지 읽어올 것인가 확인을 해야 하나?

-- 트윗데이터는 상상히 큰 용량을 차지하고 있다. 메모리에 한번에 올려서 처리하기 보다는
    나눠서 학습시킬 수 있는지 확인해 봐야겠다.
date : tweet데이터
"""

FMT_Ymd = '%Y-%m-%d'
FMT_TWEET_SEOUL = "%a %b %d %X +0900 %Y"
FMT_MINUS_JOIN = '%Y-%m-%d-%X-%z'

VECTOR_DIR_PATH = '/Users/Opi/dev/data/tweet_index_vector/vectors'
USING_MORPHEME_NUMBER = 1000  # 사용하는 형태소 수

class TweetLoader:

    def __init__(self, start='2017-04-17', end='2017-07-01'):
        self.start = start
        self.end = end

    def gen_X(self, start=None, end=None, shift=datetime.timedelta(days=0)):
        """
        
        :param start: generator가 데이터를 가져 올 시작점
        :param end: generator가 데이터를 가져 올 끝점
        :param shift: timedelta object
        :return: X, y
        :X : vecterized index vecter array
        :y : np.ndarray filled 1 or 0
        """
        if not start or not end:
            raise ValueError("gen_Xy needs '%Y-%m-%d' format string date")
        if not isinstance(shift, datetime.timedelta):
            raise TypeError('shift parameter should time delta value')

        X = []

        start = self._convert_to_date(start)
        end = self._convert_to_date(end)

        date = start
        tommorrow = datetime.timedelta(days=1)

        while date <= end:
            file_path = self._get_tweet_filepath(date) # date를 기준으로 tweet파일의 이름을 생성한다.
            if not os.path.isfile(file_path): # 해당 파일이 없으면 date 변경 후 continue 한다.
                date = date + tommorrow
                continue
            else:
                """파일이 있으면 첫 줄을 읽어서 time shift를 해 보고 몇일의 데이터인지의 정보를  X_date에 저장해 놓는다.
                그 뒤 다시 파일을 닫아서 처음부터 읽을 수 있게 한다."""
                with open(file_path, 'r') as honeymoon:
                    data = ast.literal_eval(honeymoon.readline())
                    shifted_date = datetime.datetime.strptime(data['created_at'], FMT_MINUS_JOIN) + shift
                    # X_date = shifted_date.strftime(FMT_MINUS_JOIN) # X_date 는 파일의 앞부분 날짜
                    nextday = datetime.datetime(shifted_date.year, shifted_date.month, shifted_date.day) + tommorrow
                    landmark_time = (nextday - shift).strftime(FMT_MINUS_JOIN) + '+0900'

            # created_at정보가 X_date와 비교하여 유효할 경우 append
            # 유효하지 않을 경우 stock data와 함께  yield해야 한다.
            f = open(file_path, 'r')
            for line in f:
                data = ast.literal_eval(line)
                # created_at 정보가 유효한가?
                if data['created_at'] <= landmark_time:
                    X.append(self._vectorize(data['index']))
                elif data['created_at'] > landmark_time: # only date expired it's occur
                    # yield 한다. shifted_date 변수의 날짜를 기준으로 stock data를 만든다
                    yield X, shifted_date.strftime(FMT_Ymd)
                    shifted_date = nextday
                    landmark_time = (datetime.datetime.strptime(landmark_time, FMT_MINUS_JOIN) + \
                                     tommorrow).strftime(FMT_MINUS_JOIN)
                    X = []


            if X: # if X is not empty
                yield X, shifted_date.strftime(FMT_Ymd)
                X = []

            date += tommorrow
            f.close()

    def _vectorize(self, index_list):
        if not isinstance(index_list, list):
            raise TypeError('list parameter need')

        vector = [0] * USING_MORPHEME_NUMBER
        for index in index_list:
            vector[index] = 1

        return vector


    def _convert_to_date(self, date):
        if not isinstance(date, str):
            raise TypeError('_convert_to_date needs string parameter')
        return datetime.datetime.strptime(date, FMT_Ymd)

    def _get_tweet_filepath(self, date):
        if not isinstance(date, datetime.datetime):
            raise TypeError('_get_tweet_filepath needs datetime parameter')
        # file nama example : vector-2017-04-19.txt
        filename = 'vector-' + date.strftime(FMT_Ymd) + '.txt'
        return os.path.join(VECTOR_DIR_PATH, filename)


if __name__ == '__main__':
    loader = TweetLoader()
    for X, date in loader.gen_X('2017-04-17', '2017-04-30', shift=datetime.timedelta(minutes=210)):
        a = 1
        X
        date

