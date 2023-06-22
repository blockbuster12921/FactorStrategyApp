import unittest
import logging
import sys
import pandas as pd
import numpy as np
import datetime

if ('-v' in sys.argv) or ('--verbose' in sys.argv):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

from FactorCore import data_io

class TestDataIO(unittest.TestCase):

    def test_load_from_excel(self):

        loader = data_io.DataLoader()

        loader.load_from_excel("/data/Industrials Data Upload Minimal.xlsx")

        self.assertEqual(loader.dates[0], pd.Timestamp('2004-01-31'))
        self.assertEqual(loader.dates[-1], pd.Timestamp('2020-01-31'))

        self.assertEqual(len(loader.stocks), 66)
        for stock in loader.stocks:
            self.assertGreater(len(stock['Name']), 0)
            self.assertGreater(len(stock['Ticker']), 0)
            self.assertGreater(len(stock['SubSector']), 0)
        self.assertEqual(loader.stocks[0]['Name'], 'United Technologies Corp')
        self.assertEqual(loader.stocks[0]['Ticker'], 'UTX US Equity')
        self.assertEqual(loader.stocks[0]['SubSector'], 'Aerospace & Defense')

        self.assertEqual(loader.factor_names, ['F1','F2','F3','F4','F5','F6','F8','F9','F10','F36','F11','F12','F25','F28','F37','F38','F40','F41','F42','F43','F7'])

        self.assertEqual(len(loader.stocks), len(loader.returns_df.columns))
        self.assertEqual(returns_df.columns.to_list(), range(0,len(loader.stocks)))
        self.assertEqual(len(loader.dates), len(loader.returns_df.index))

        self.assertEqual(len(loader.stocks), len(loader.factors_df.columns))
        self.assertEqual(factors_df.columns.to_list(), range(0,len(loader.stocks)))
        self.assertEqual(len(loader.factor_names)*len(loader.dates), len(loader.factors_df.index))
        self.assertEqual(len(loader.dates), len(loader.factors_df.loc[0,:].index))
