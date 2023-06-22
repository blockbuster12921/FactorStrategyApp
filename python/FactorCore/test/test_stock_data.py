import unittest
import logging
import sys
import pandas as pd
import numpy as np
import datetime

if ('-v' in sys.argv) or ('--verbose' in sys.argv):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

from FactorCore import stock_data

class TestStockData(unittest.TestCase):

    def test_load_from_excel(self):

        sd = stock_data.StockData()

        self.assertTrue(sd.is_empty())
        self.assertEqual(sd.get_date_range(), (None, None))
        self.assertEqual(sd.get_factor_indexes(), [])
        self.assertEqual(sd.get_factor_names(), [])

        sd.load_from_excel("/data/Factor Upload Sheet Minimal.xlsx")

        self.assertFalse(sd.is_empty())
        self.assertEqual(sd.returns_df.columns[0], sd.factors_df[0].columns[0])
        self.assertEqual(sd.returns_df.columns[-1], sd.factors_df[0].columns[-1])
        self.assertEqual(sd.get_date_range()[0].year, 2004)
        self.assertEqual(sd.get_date_range()[0].month, 1)
        self.assertEqual(sd.get_date_range()[1].year, 2019)
        self.assertEqual(sd.get_date_range()[1].month, 12)

        stock_names = sd.get_stock_names()
        self.assertEqual(len(stock_names), 69)
        self.assertEqual(len(stock_names), len(sd.returns_df.index))
        self.assertEqual(sd.factors_df.index.to_list(), sd.returns_df.index.to_list())

        self.assertEqual(sd.get_factor_indexes(), [0,1,2,3])
 
        factor_names = sd.get_factor_names()
        self.assertEqual(len(factor_names), 4)
        self.assertTrue("F1" in factor_names)
        self.assertTrue("F2" in factor_names)
        self.assertTrue("F3" in factor_names)
        self.assertTrue("F4" in factor_names)

        # Shrink date range
        sd.shrink_date_range(('2010-01-31', '2017-06-30'))
        self.assertEqual(sd.returns_df.columns[0], sd.factors_df[0].columns[0])
        self.assertEqual(sd.returns_df.columns[-1], sd.factors_df[0].columns[-1])
        self.assertEqual(sd.get_date_range()[0].year, 2010)
        self.assertEqual(sd.get_date_range()[0].month, 1)
        self.assertEqual(sd.get_date_range()[1].year, 2017)
        self.assertEqual(sd.get_date_range()[1].month, 6)
