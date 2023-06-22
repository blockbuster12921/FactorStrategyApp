import unittest
import logging
import sys
import pandas as pd
import numpy as np
import datetime

if ('-v' in sys.argv) or ('--verbose' in sys.argv):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

from FactorCore import generate, stock_data

class TestGenerate(unittest.TestCase):

    def test_generate(self):

        # Load data
        sd = stock_data.StockData()
        sd.load_from_excel("/data/Factor Upload Sheet Minimal.xlsx")
        self.assertEqual(sd.get_factor_indexes(), [0,1,2,3])

        sd.shrink_date_range((pd.Timestamp('2010-12-31'), pd.Timestamp('2019-11-30')))
        self.assertEqual(sd.get_factor_indexes(), [0,1,2,3])

        # Generate
        gen = generate.Generator(
            cache=None, 
            data=sd,
            st_months=6,
            st_weight=1.0,
            score_pairs_range=(6,10),
            factor_count_range=(1,4),
            max_combinations=10,
            )

        factors, scores = gen._generate_for_base([])
        self.assertEqual(factors, [3, 1, 2, 0])
        for f in factors:
            self.assertTrue(not np.isnan(scores[f]))
