import pandas as pd
import numpy as np
import pyarrow
import logging
from typing import List, Tuple
import timeit

logger = logging.getLogger(__name__)

# Stock factors and returns for a sector
class StockData:

    def __init__(self):
        self.factors_df = pd.DataFrame()
        self.returns_df = pd.DataFrame()
        self.factor_names = []

    def is_empty(self) -> bool:
        return self.factors_df.empty or self.returns_df.empty

    def get_stock_names(self) -> List[str]:
        return self.returns_df.index.to_list()

    def get_factor_indexes(self) -> List[int]:
        if self.factors_df.empty:
            return []
        return self.factors_df.columns.levels[0].to_list()

    def get_factor_names(self) -> List[str]:
        return self.factor_names

    def get_date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        if self.returns_df.empty:
            return (None, None)
        return (self.returns_df.columns[0], self.returns_df.columns[-1])

    def shrink_date_range(self, new_date_range):
        self.returns_df = self.returns_df.T.loc[new_date_range[0]:new_date_range[1]].T

        slice_idx = pd.MultiIndex.from_product([self.get_factor_indexes(), self.returns_df.columns])        
        self.factors_df = self.factors_df.T.loc[slice_idx].T

    def load_from_excel(self, data_file):

        # Read from Excel
        start_time = timeit.default_timer()
        logger.info("Reading data from Excel file...")
        sheet = pd.read_excel(data_file, sheet_name=0, header=None, convert_float=False)
        logger.info("read_excel complete in {:.3f} seconds".format(timeit.default_timer() - start_time))
        
        # Drop blank columns
        sheet = sheet.dropna(axis=1, how='all')

        if len(sheet.index) < 95:
            raise ValueError("Not enough rows in worksheet")
        if len(sheet.columns) < 2:
            raise ValueError("Not enough columns in worksheet")

        # Read returns
        
        # Skip blank rows
        def skip_blank_rows(sheet, row_index):
            while True:
                if row_index == len(sheet.index):
                    return row_index
                if not sheet.iloc[row_index].isnull().all():
                    return row_index
                row_index += 1

        row_index = skip_blank_rows(sheet, 0)
        if row_index == len(sheet.index):
            raise ValueError("No non-blank rows in data")

        # First row is name followed by dates
        
        # Check that name contains "return"
        row = sheet.iloc[row_index]
        if "return" not in row.iloc[0].lower():
            raise ValueError("First non-blank cell in row {} ('{}') does not contain 'Return'".format(row_index+1, row.iloc[0]))

        # Read dates
        dates_data = pd.to_datetime(row.iloc[1:])
        dates = pd.date_range(dates_data.iloc[0], periods=len(dates_data.index), freq='m')
        for date_index in range(0, len(dates)):
            d1 = dates_data.iloc[date_index]
            d2 = dates[date_index]
            if (d1.year != d2.year) or (d1.month != d2.month):
                raise ValueError("Found date {:%b-%Y} where {:%b-%Y} was expected".format(d1, d2))
        
        # Read stock names and return values
        returns_df = pd.DataFrame(index=dates)
        returns_df.index.names = ['Date']

        row_index += 1
        while row_index < len(sheet.index):
            row = sheet.iloc[row_index]
            if row.isnull().all():
                row_index += 1
                break
                
            stock = row.iloc[0]
            if len(stock) == 0:
                raise ValueError("Blank stock name in row {}".format(row_index+1))

            returns_df[stock] = pd.to_numeric(row.iloc[1:].values, errors='coerce')

            row_index += 1

        if len(returns_df.columns) == 0:
            raise ValueError("No returns data found")
            
        logger.info("Returns data read successfully: found {} stocks in date range {:%b-%Y} to {:%b-%Y}".format(
            len(returns_df.columns), dates[0], dates[-1]))

        row_index = skip_blank_rows(sheet, row_index)
        if row_index == len(sheet.index):
            raise ValueError("No Factors found")

        def read_factor(row_index, returns_df):

            row = sheet.iloc[row_index]

            # Factor name is first value of row
            factor_name = row.iloc[0]
            if pd.isnull(factor_name):
                logger.info("Blank Factor name in row {} - skipping factor".format(row_index+1))
                return (None, None)

            logger.info("Reading data for Factor '{}'".format(factor_name))

            # Read stock names and factor values
            factor_df = pd.DataFrame(index=dates)
            factor_df.index.names = ['Date']
            
            for stock_index, stock in enumerate(returns_df.columns):

                row_index += 1
                if row_index == len(sheet.index):
                    raise ValueError("Too few stocks for Factor '{}'".format(factor_name))
                
                row = sheet.iloc[row_index]
                if row.isnull().all():
                    raise ValueError("Blank row {} when Factor values for stock expected".format(row_index+1))

                factor_stock = row.iloc[0]
                if factor_stock != stock:
                    raise ValueError("Found stock name '{}' on row {} for Factor '{}' - expected '{}'".format(
                        factor_stock, row_index+1, factor_name, stock))

                factor_df[stock] = pd.to_numeric(row.iloc[1:].values, errors='coerce')

            if factor_df.isnull().all().all():
                logger.info("No valid values for Factor '{}' - skipping factor".format(factor_name))
                return (None, None)

            for date in returns_df.index:
                if returns_df.loc[date].count() < factor_df.loc[date].count():
                    raise ValueError("Factor '{}' has more non-blank values ({}) than Returns ({}) for {:%b-%Y}".format(
                        factor_name, factor_df.loc[date].count(), returns_df.loc[date].count(), date))

            return (factor_df, factor_name)

        # Read factors
        factor_df_list = []
        factor_names = []
        while True:
            factor_df, factor_name = read_factor(row_index, returns_df)

            if factor_df is not None:
                
                if factor_name in factor_names:
                    raise ValueError("Factor name '{}' occurs more than once".format(factor_name))
                
                factor_df.insert(0, 'Factor', len(factor_df_list))
                factor_df_list.append(factor_df.reset_index())
                factor_names.append(factor_name)
                logger.info("Factor '{}' read successfully".format(factor_name))

            row_index += 95
            if row_index >= len(sheet.index):
                break

        # Finish up
        self.returns_df = returns_df
        self.factor_names = factor_names
        self.factors_df = pd.concat(factor_df_list, sort=False).set_index('Date')
        
        logger.info("Read {} factors".format(len(factor_df_list)))
        if len(factor_df_list) < 1:
            raise ValueError("Data must contain one or more factors")

        # Find first and last dates where we have one or more non-null returns value
        returns_value_count = self.returns_df.notnull().sum(axis=1)
        valid_returns_dates = returns_value_count.loc[returns_value_count > 0].index
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

        self.factors_df = self.factors_df.reset_index().set_index(['Factor','Date']).T
        self.returns_df = self.returns_df.T

        logger.info("Loaded returns and {} factors for {} stocks".format(len(self.factors_df.columns.get_level_values('Factor').unique()), len(self.factors_df.index)))


    def serialize(self):
        pyarrow_context = pyarrow.default_serialization_context()
        return {
            'Returns': pyarrow_context.serialize(self.returns_df).to_buffer().to_pybytes(),
            'Factors': pyarrow_context.serialize(self.factors_df).to_buffer().to_pybytes(),
            'FactorNames': pyarrow_context.serialize(self.factor_names).to_buffer().to_pybytes(),
        }

    def deserialize(self, serialization):
        pyarrow_context = pyarrow.default_serialization_context()
        self.returns_df = pyarrow_context.deserialize(serialization['Returns'])
        self.factors_df = pyarrow_context.deserialize(serialization['Factors'])
        self.factor_names = pyarrow_context.deserialize(serialization['FactorNames'])


