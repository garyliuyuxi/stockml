import logging
import numpy
import pandas
import quandl
import sys
import tushare

from datetime import datetime
from pandas_datareader import data as web
from util import Util
from alpha_vantage.timeseries import TimeSeries

class Data:

    def __init__(self,
                symbols,
                start_date,
                end_date,
                raw_data_directory,
                output_directory,
                logger=None):

        self.util = Util.Instance()
        
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger("ML."+__name__)
        
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.raw_data_directory = raw_data_directory
        self.output_directory = output_directory

        # Constants
        self.COLS_RAW = ['timestamp', 'id', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
        self.COLS_DATA = ['open', 'high', 'low', 'close', 'volume', 'adj_close']
        self.IDX_DATE = 0
        self.OL_TARGET = 'open'
        self.COL_ADJ = 'adj_close'
        self.COLS_TRAIN_EXCLUDE = ['open', 'high', 'low', 'close', 'volume', 'adj_close', 'y']
        self.FIELD_OUTPUT_CSV = ('timestamp', 'id', 'open', 'high', 'low', 'close', 'volume', 'adj_close', 'y', 'y_pred')
        self.ML_MODEL_LIST = ['rfr', 'gbr']
        self.tushare_data_renamed_columns = {'open': 'adj_open',
                                                'high': 'adj_high',
                                                'low': 'adj_low',
                                                'close': 'adj_close'}
        self.tushare_data_dropped_columns = ['volume', 'price_change', 'p_change', 'ma5', 'ma10', 'ma20', 'v_ma5', 'v_ma10', 'v_ma20']
        self.tushare_data_columns = ['adj_open', 'adj_high', 'adj_low', 'adj_close']

        self.alpha_vantage_data_renamed_columns = {'1. open': 'adj_open',
                                                    '2. high': 'adj_high',
                                                    '3. low': 'adj_low',
                                                    '4. close': 'adj_close',
                                                    'date': 'timestamp'}
        self.alpha_vantage_data_dropped_columns = ['5. volume']

        self.yahoo_data_renamed_columns = {'Adj. Open': 'adj_open',
                                            'Adj. High': 'adj_high',
                                            'Adj. Low': 'adj_low',
                                            'Adj. Close': 'adj_close',
                                            'Date': 'timestamp'}
        self.yahoo_data_dropped_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj. Volume']
        
        self.quandl_data_renamed_columns = {'Adj. Open': 'adj_open',
                                            'Adj. High': 'adj_high',
                                            'Adj. Low': 'adj_low',
                                            'Adj. Close': 'adj_close'}
        self.quandl_data_dropped_columns = ['Ex-Dividend', 'Split Ratio', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj. Volume']

        # raw data dictionaries with symbol and dataframes
        self.df_dict = {}

    def get_chinese_stock_info(self):
        self.logger.info("Get Chinese stock information")
        df = tushare.get_stock_basics()
        df.to_csv("{0}/chinese_stocks.csv".format(self.output_directory))
        

    def fetch_tushare_data(self):
        self.logger.info("\tFetching data from tushare")

        # Convert start_date and end_date to datetime objects
        start_date = datetime.strptime(self.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(self.end_date, '%Y-%m-%d')

        for s in self.symbols:
            try:
                self.logger.info("\t\tFetch raw data for {0}".format(s))
                df = tushare.get_hist_data(s)
                df.index = pandas.to_datetime(df.index, format='%Y-%m-%d').normalize()
                df = df.loc[(df.index > start_date) & (df.index <= end_date)]
                self.logger.debug("\n{0} raw data\n{1}".format(s, df))
            except:
                raise RuntimeError("\tCan not fetch data for {0} from tushare".format(s))
            df.drop(columns=self.tushare_data_dropped_columns, inplace=True)
            df.rename(columns=self.tushare_data_renamed_columns, inplace=True)
            df.index.names = ['timestamp'] # name the index column
            df = df[self.tushare_data_columns]
            df.insert(0, 'id', s)
            df.sort_index(inplace=True)

            self.logger.debug("\ndf shape = {0}".format(df.shape))
            self.logger.debug("\n{0} raw data for save\n{1}".format(s, df))

            # save the data to csv file
            self.util.check_dir_exist(self.output_directory)
            df.to_csv("{0}/{1}.csv".format(self.output_directory, s))

            # save the data in Dataframe class raw data dataframes
            self.df_dict[s] = df

        self.align_timestamps()
        

    def fetch_alpha_vantage_data(self):
        """

        :param symbol:
        :param t_start:
        :param t_end:
        :param outdir:
        :return:
        """
        self.logger.info("\tDatafetch - fetching from Alpha Vantage")
        self.logger.info("\tAuthenticate")
        ts = TimeSeries(key='4EHF2EYGFIK6YIHI', output_format='pandas', indexing_type='date')

        # Convert start_date and end_date to datetime objects
        start_date = datetime.strptime(self.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(self.end_date, '%Y-%m-%d')

        for symbol in self.symbols:
            self.logger.info("\tFetch raw data for {0}".format(symbol))
            try:
                df, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
                df.index = pandas.to_datetime(df.index, format='%Y-%m-%d').normalize() # remove 00:00:00 at the end of the date
                df = df.loc[(df.index > start_date) & (df.index <= end_date)]
            except:
                raise RuntimeError("Can not fetch data from Alpha Vantage")

            df.drop(columns=self.alpha_vantage_data_dropped_columns, inplace=True)
            df.rename(columns=self.alpha_vantage_data_renamed_columns, inplace=True)
            df.index.names = ['timestamp']
            df.insert(0, 'id', symbol)

            self.logger.debug("\ndf shape = {0}".format(df.shape))
            self.logger.debug("\n{0} raw data for save\n{1}".format(symbol, df))

            self.util.check_dir_exist(self.output_directory)
            df.to_csv("{0}/{1}.csv".format(self.output_directory, symbol))

            # save the data in Dataframe class raw data dataframes
            self.df_dict[symbol] = df
            
        self.align_timestamps()

        return 0

    def fetch_quandl_data(self):
        """

        :param symbol:
        :param t_start:
        :param t_end:
        :param outdir:
        :return:
        """
        self.logger.info("\tDatafetch - fetching from Quandl")
        self.logger.info("\tAuthenticate")

        quandl.ApiConfig.api_key = "MrKQAcw9ToRePskdgfgG"

        # Convert start_date and end_date to datetime objects
        start_date = datetime.strptime(self.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(self.end_date, '%Y-%m-%d')

        for symbol in self.symbols:
            self.logger.info("\tFetch raw data for {0}".format(symbol))
            try:
                data = quandl.get('WIKI/{0}'.format(symbol), start_date=start_date, end_date=end_date)
            except:
                raise RuntimeError("Can not fetch data from Quandl")

            df_raw_data = pandas.DataFrame(data=data)

            df_raw_data.index = pandas.to_datetime(df_raw_data.index, format='%Y-%m-%d').normalize() # remove 00:00:00 at the end of the date
            df_raw_data.drop(columns=self.quandl_data_dropped_columns, inplace=True)
            df_raw_data.rename(columns=self.quandl_data_renamed_columns, inplace=True)
            df_raw_data.index.names = ['timestamp']
            df_raw_data.insert(0, 'id', symbol)

            self.logger.debug("\ndf shape = {0}".format(df_raw_data.shape))
            self.logger.debug("\n{0} raw data for save\n{1}".format(symbol, df_raw_data))

            self.util.check_dir_exist(self.output_directory)
            df_raw_data.to_csv("{0}/{1}.csv".format(self.output_directory, symbol))

            # save the data in Dataframe class raw data dataframes
            self.df_dict[symbol] = df_raw_data
            
        self.align_timestamps()

    def fetch_yahoo_data(self, symbols, t_start, t_end, output_directory):
        """
        :param symbols: a list of symbols
        :param t_start:
        :param t_end:
        :param output_directory: the directory data downloads to
        :return:
        """

        start = datetime.strptime(t_start, '%Y-%m-%d')
        end = datetime.strptime(t_end, '%Y-%m-%d')

        for symbol in symbols:
            print("fetch raw data from Yahoo for {0}".format(symbol))

            for i in range(10):
                try:
                    df_raw_data = web.get_data_yahoo(symbol, start, end)
                    break
                except:
                    if i < 9:
                        print("retry")
                    else:
                        raise RuntimeError("Still can not fetch data from Yahoo after {0} attempts".format(i+1))
                    
            df_raw_data = df_raw_data.rename(index=str, columns=self.yahoo_data_renamed_columns)
            df_raw_data.index.names = ['timestamp']
            df_raw_data.insert(0, 'id', symbol)

            self.util.check_dir_exist(output_directory)
            df_raw_data.to_csv("{0}/{1}.csv".format(output_directory, symbol))

        self.align_timestamps(output_directory, symbols)

    def align_timestamps(self):
        self.logger.info("\tAlign timestamps")

        index_array = []

        self.logger.info("\t\tFind common timestamps")
        for s in self.symbols:
            index_array.append(self.df_dict[s].index.values)
            self.logger.debug("\nsymbol - {0}\n{1}".format(s, self.df_dict[s]))

        # Find the intersection of indexes of all items
        index = set(index_array[0])
        for i in index_array[1:]:
            index.intersection_update(i)
        
        # Reduce the timestamps down to the minimum so that timestamps of all items are aligned 
        self.logger.info("\tRemove data with timestamps not in common")

        for s in self.symbols:
            self.df_dict[s] = self.df_dict[s].loc[index, :].sort_index()

            self.logger.debug("\ndf index\n{0}".format(self.df_dict[s].index))
            self.logger.debug("\ndf shape = {0}".format(self.df_dict[s].shape))
            # self.logger.info("\ndf\n{0}".format(df))
            self.df_dict[s].to_csv("{0}/{1}.csv".format(self.raw_data_directory, s))





