import pandas as pd
from typing import List, Tuple
import timeit
import logging

logger = logging.getLogger(__name__)

class DataLoader:

    def __init__(self):
        self.sheet = None
        self.row_index = None
        self.dates = None
        self.stocks = []
        self.returns_df = None
        self.factor_df_list = []
        self.factor_names = []
        self.factors_df = None

    def load_from_excel(self, data_file):
        
        self._load_excel_sheet(data_file)
        
        self._read_data()
    
    def _load_excel_sheet(self, data_file):

        # Read from Excel
        start_time = timeit.default_timer()
        logger.info("Reading data from Excel file...")
        sheet = pd.read_excel(data_file, sheet_name=0, header=None, convert_float=False)
        logger.info("read_excel complete in {:.3f} seconds".format(timeit.default_timer() - start_time))
        
        # Drop blank columns
        sheet = sheet.dropna(axis=1, how='all')

        if len(sheet.index) < 95:
            raise ValueError("Not enough rows in worksheet")
        if len(sheet.columns) < 3:
            raise ValueError("Not enough columns in worksheet")
            
        self.sheet = sheet

    # Skip blank rows
    def _skip_blank_rows(self):
        while True:
            if self.row_index == len(self.sheet.index):
                return
            if not self.sheet.iloc[self.row_index].isnull().all():
                return
            self.row_index += 1
            
    def is_end_of_sheet(self):
        return self.row_index >= len(self.sheet.index)

    def _parse_dates(self, dates_data):

        try:
            first_date = pd.to_datetime(dates_data[0])
        except Exception:
            raise ValueError("First date '{}' is not in a valid date format".format(dates_data[0]))

        self.dates = pd.date_range(first_date, periods=len(dates_data), freq='m')
        for i, d1 in enumerate(dates_data):
            try:
                d1_as_date = pd.to_datetime(d1)
            except Exception:
                raise ValueError("Date '{}' ({} of {}) is not in a valid date format".format(d1, i+1, len(dates_data)))

            d2 = self.dates[i]

            if (d1_as_date.year != d2.year) or (d1_as_date.month != d2.month):
                raise ValueError("Found '{}' where date {:%b-%Y} was expected (date {} of {})".format("" if pd.isnull(d1) else d1, d2, i+1, len(dates_data)))

    def _read_stocks_and_returns(self):
        
        self.stocks = []
        self.returns_df = pd.DataFrame(index=self.dates)
        self.returns_df.index.names = ['Date']

        while self.row_index < len(self.sheet.index):
            row = self.sheet.iloc[self.row_index]
            if row.isnull().all():
                self.row_index += 1
                break

            stock_subsector = row.iloc[0]
            if len(stock_subsector) == 0:
                raise ValueError("Blank stock sub-sector in row {}".format(self.row_index+1))

            stock_ticker = row.iloc[1]
            if len(stock_ticker) == 0:
                raise ValueError("Blank stock ticker in row {}".format(self.row_index+1))

            stock_name = row.iloc[2]
            if len(stock_name) == 0:
                raise ValueError("Blank stock name in row {}".format(self.row_index+1))

            self.returns_df[len(self.stocks)] = pd.to_numeric(row.iloc[3:].values, errors='coerce')

            self.stocks.append({'Name': stock_name, 'Ticker': stock_ticker, 'SubSector': stock_subsector})

            self.row_index += 1

        if len(self.returns_df.columns) == 0:
            raise ValueError("No returns data found")

        logger.info("Returns data read successfully: found {} stocks in date range {:%b-%Y} to {:%b-%Y}".format(
            len(self.returns_df.columns), self.dates[0], self.dates[-1]))

    def _read_factor(self):

        row = self.sheet.iloc[self.row_index]

        # Factor name is third value of row
        factor_name = row.iloc[2]
        if pd.isnull(factor_name):
            logger.info("Blank Factor name in row {} - skipping row".format(self.row_index+1))
            self.row_index += 1
            return (None, None)

        if factor_name == self.stocks[0]['Name']:
            logger.info("Factor name = first stock name - skipping {} rows".format(len(self.stocks)))
            self.row_index += len(self.stocks)
            return (None, None)

        logger.info("Reading data for Factor '{}'".format(factor_name))

        # Read stock names and factor values
        factor_df = pd.DataFrame(index=self.dates)
        factor_df.index.names = ['Date']

        for stock_index in self.returns_df.columns:

            self.row_index += 1
            if self.is_end_of_sheet():
                raise ValueError("Too few stocks for Factor '{}'".format(factor_name))

            row = self.sheet.iloc[self.row_index]
            if row.isnull().all():
                raise ValueError("Blank row {} when Factor values for stock expected".format(self.row_index+1))

            factor_stock = row.iloc[2]
            stock_name = self.stocks[stock_index]['Name']
            if factor_stock != stock_name:
                raise ValueError("Found stock name '{}' on row {} for Factor '{}' - expected '{}'".format(
                    factor_stock, self.row_index+1, factor_name, stock_name))

            factor_df[stock_index] = pd.to_numeric(row.iloc[3:].values, errors='coerce')

        self.row_index += 1
            
        if factor_df.isnull().all().all():
            logger.info("No valid values for Factor '{}' - skipping factor".format(factor_name))
            return (None, None)

        return (factor_df, factor_name)

    def _read_factors(self):

        # Skip blank rows
        self._skip_blank_rows()
        if self.is_end_of_sheet():
            raise ValueError("No Factors found")

        # Read factors
        self.factor_df_list = []
        self.factor_names = []
        while True:
            factor_df, factor_name = self._read_factor()

            if factor_df is not None:
                
                if factor_name in self.factor_names:
                    raise ValueError("Factor name '{}' occurs more than once".format(factor_name))

                factor_df.insert(0, 'Factor', len(self.factor_df_list))
                self.factor_df_list.append(factor_df.reset_index())
                self.factor_names.append(factor_name)
                logger.info("Factor '{}' read successfully".format(factor_name))

            self._skip_blank_rows()
            if self.is_end_of_sheet():
                break

        logger.info("Read {} factors".format(len(self.factor_df_list)))
        if len(self.factor_df_list) < 1:
            raise ValueError("Data must contain one or more factors")


    def _read_data(self):
        
        self.row_index = 0

        # Read returns
        self._skip_blank_rows()
        if self.is_end_of_sheet():
            raise ValueError("No non-blank rows in data")

        # First row is three headers followed by dates
        # Check that third header contains "return"
        row = self.sheet.iloc[self.row_index]
        if len(row) < 5:
            raise ValueError("Not enough non-blank cells in first non-blank row {}".format(self.row_index+1))
        if "return" not in row.iloc[2].lower():
            raise ValueError("Third non-blank cell in row {} ('{}') does not contain 'Return'".format(self.row_index+1, row.iloc[2]))

        # Read dates
        self._parse_dates(row.iloc[3:].values)

        # Read stock sub-sector, tickers, names and return values
        self.row_index += 1
        self._read_stocks_and_returns()

        # Read factors
        self._read_factors()
        self.factors_df = pd.concat(self.factor_df_list, sort=False).set_index('Date')

        # Find first and last dates where we have one or more non-null returns value (except for last month)
        returns_value_count = self.returns_df.count(axis=1)
        valid_returns_dates = returns_value_count.loc[(returns_value_count > 0) | (returns_value_count.shift(1) > 0)].index
        if len(valid_returns_dates) == 0:
            raise ValueError("No dates found with valid returns data")

        first_valid_date = valid_returns_dates[0]
        last_valid_date = valid_returns_dates[-1]

        # Find first and last dates where we have one or more non-null factor values
        factor_value_count = self.factors_df.notnull().sum(axis=1) - 1
        factor_value_count = factor_value_count.loc[factor_value_count.index >= first_valid_date]
        factor_value_count = factor_value_count.loc[factor_value_count.index <= last_valid_date]

        first_valid_date = None
        for index, count in factor_value_count.iteritems():
            if count > 0:
                first_valid_date = index
                break

        last_valid_date = None
        for index, count in factor_value_count[::-1].iteritems():
            if count > 0:
                last_valid_date = index
                break

        if (first_valid_date is None) or (last_valid_date is None):
            raise ValueError("No dates found with valid factor data")

        logger.info("Date range for valid data is {:%b-%Y} to {:%b-%Y}".format(first_valid_date, last_valid_date))        

        self.factors_df = self.factors_df.loc[(self.factors_df.index >= first_valid_date) & (self.factors_df.index <= last_valid_date)]
        self.returns_df = self.returns_df.loc[(self.returns_df.index >= first_valid_date) & (self.returns_df.index <= last_valid_date)]
        self.dates = self.returns_df.index.to_list()
        
        self.factors_df = self.factors_df.reset_index().set_index(['Factor','Date'])

        logger.info("Loaded returns and {} factors for {} stocks".format(len(self.factor_names), len(self.stocks)))

