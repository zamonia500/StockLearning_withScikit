import os
import ast
import numpy as np
import datetime
import glob

DIR_PATH = '/Users/Opi/dev/data/stockData/data_1.1' # kospi added, fixed invalid value

class StockLoader:
    """
    set_nth_batch : self.stock_batch를 설정
    """

    def __init__(self, dir_path , batch_size=50):
        """
        :param dir_path: stock date file이 있는 폴더의 경로
        :param batch_size: self.stock_batch의 크기
        """
        self.stock = []
        self.batch_size = batch_size
        self.dir_path = dir_path
        self._get_stock_filelist(dir_path)
        self._make_stock_diclist()
        #self._make_file_iter()

    def get_one_stock(self, stock_name):
        stock = {}

        full_filepath = os.path.join(self.dir_path, stock_name)
        with open(full_filepath, 'r') as f:
            for line in f:
                data = ast.literal_eval(line)
                stock[data['created_at']] = data['stock']

        return stock

    def get_dic_for_predict(self):
        """
        폴더 안의 모든 stock data를 읽어서 recursice dictionary를 만들어 리턴해준다.
        :return: 
        """
        if not os.path.isdir(DIR_PATH):
            raise ValueError('dir_path not exist or invalid path')

        full_dic = {}
        stock_files = glob.glob(os.path.join(DIR_PATH, '*.txt'))

        for stock_file in stock_files: # 모든 파일에 대해서 반복
            stock_dic = {}
            with open(stock_file) as f:
                for line in f:
                    data = ast.literal_eval(line)
                    stock_dic[data['created_at']] = data['stock']
            full_dic[os.path.basename(stock_file)] = stock_dic

        return full_dic

    def set_nth_batch(self, n):
        if not isinstance(n, int):
            raise TypeError("param n must integer")
        if n * self.batch_size >= len(self.stock):
            raise OverflowError("invalid batch number")

        self.stock_batch = self.stock[n*self.batch_size : (n+1)*self.batch_size]

    def get_batch_stock(self, index, date, length):
        for _ in range(15): # 만약 원하는 날짜가 주말또는 공휴일이라 값이 존재하지 않으면 다음날을 찾아본다.
            if date in self.stock_batch[index][1]:
                value = self.stock_batch[index][1][date]
                break
            else:
                vars = list(map(int, date.split('-')))
                next_date = datetime.datetime(vars[0], vars[1], vars[2]) + datetime.timedelta(days=1)
                date = '{0}-{1:02d}-{2:02d}'.format(next_date.year,next_date.month,next_date.day)
        arr = np.ndarray(shape=(length,), dtype=float)
        arr.fill(value)

        return arr

    def get_y(self, index, y_source):
        if not isinstance(y_source, list):
            raise TypeError('y_source : [(date1, len(X1)), (date2, len(X2), (date3, len(X2)], ....')

        y = np.ndarray(shape=(0,), dtype=float)
        for date, length in y_source:
            for _ in range(15):  # 만약 원하는 날짜가 주말또는 공휴일이라 값이 존재하지 않으면 다음날을 찾아본다.
                if date in self.stock_batch[index][1]:
                    value = self.stock_batch[index][1][date]
                    break
                else:
                    vars = list(map(int, date.split('-')))
                    next_date = datetime.datetime(vars[0], vars[1], vars[2]) + datetime.timedelta(days=1)
                    date = '{0}-{1:02d}-{2:02d}'.format(next_date.year, next_date.month, next_date.day)
            arr = np.ndarray(shape=(length,), dtype=float)
            arr.fill(value)
            y = np.append(y, arr)

        return y

    def get_stock_arr(self, date, length):
        """
        
        :param date: 몇 일
        :param length: 길이 몇
        :return: 
        """
        for _ in range(15): # 만약 원하는 날짜가 주말또는 공휴일이라 값이 존재하지 않으면 다음날을 찾아본다.
            if date in self.stock_data:
                value = self.stock_data[date]
                break
            else:
                vars = list(map(int, date.split('-')))
                next_date = datetime.datetime(vars[0], vars[1], vars[2]) + datetime.timedelta(days=1)
                date = '{0}-{1:02d}-{2:02d}'.format(next_date.year,next_date.month,next_date.day)
        arr = np.ndarray(shape=(length,), dtype=float)
        arr.fill(value)

        return arr

    def get_stock(self, index, date, length):
        for _ in range(15): # 만약 원하는 날짜가 주말또는 공휴일이라 값이 존재하지 않으면 다음날을 찾아본다.
            if date in self.stock[index][1]:
                value = self.stock[index][1][date]
                break
            else:
                vars = list(map(int, date.split('-')))
                next_date = datetime.datetime(vars[0], vars[1], vars[2]) + datetime.timedelta(days=1)
                date = '{0}-{1:02d}-{2:02d}'.format(next_date.year,next_date.month,next_date.day)
        arr = np.ndarray(shape=(length,), dtype=float)
        arr.fill(value)

        return arr

    def make_next_stock_dic(self):
        """
        self.iter_filelist 에서 파일 이름을 가져온다.
        
        :return: 
        """
        self.filename = next(self.iter_filelist)
        full_filepath = os.path.join(self.stock_dir_path, self.filename)
        stock = {}

        with open(full_filepath, 'r') as f:
            for line in f:
                try:
                    data = ast.literal_eval(line)
                    stock[data['created_at']] = data['stock']
                except Exception as e:
                    print(str(e))
        self.stock_data = stock

        return self.stock_data

    def _get_stock_filelist(self, dir_path):
        """        
        :param dir_path: the directory contains stock info files
        :return: len(list(stock file list))
        """
        if not os.path.isdir(dir_path):
            raise ValueError('invalid dir_path: there is no such dir')

        stock_filelist = []
        filenames = os.listdir(dir_path)

        for filename in filenames:
            full_filename = os.path.join(dir_path, filename)
            ext = os.path.splitext(full_filename)[-1]
            if ext == '.txt':
                stock_filelist.append(filename)

        self.stock_dir_path = dir_path
        self.stock_filelist = stock_filelist

        return len(stock_filelist)

    def _make_stock_diclist(self):
        """
        list = [(filename, stock_dictionary) ~] 로 다 채워넣자
        
        :return: 
        """
        while self.stock_filelist:
            stock = {}

            file = self.stock_filelist.pop()
            full_filepath = os.path.join(self.stock_dir_path, file)
            with open(full_filepath, 'r') as f:
                for line in f:
                        try:
                            data = ast.literal_eval(line)
                            stock[data['created_at']] = data['stock']
                        except Exception as e:
                            print(str(e))

            self.stock.append((file, stock))

        return len(self.stock)

    def _make_file_iter(self):
        self.iter_filelist = iter(self.stock_filelist)


if __name__ == '__main__':
    st_loader = StockLoader(DIR_PATH)
    file , stock = st_loader.stock[1]
    print(file)

    '''
    st_loader.make_next_stock_dic()
    print(st_loader.filename)
    print(st_loader.get_stock_arr('2017-04-05', 100))
    print(st_loader.get_stock_arr('2017-05-01', 50))
    print(st_loader.get_stock_arr('2017-06-04', 125))
    st_loader.make_next_stock_dic()
    print(st_loader.filename)
    print(st_loader.get_stock_arr('2017-07-05', 25))
    print(st_loader.get_stock_arr('2017-06-13', 64))
    print(st_loader.get_stock_arr('2017-05-15', 100))
    '''
