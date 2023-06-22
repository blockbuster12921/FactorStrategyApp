import unittest
import logging
import sys
import pandas as pd
import numpy as np
import datetime

if ('-v' in sys.argv) or ('--verbose' in sys.argv):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

from FactorCore import optimize, stock_data

class TestOptimize(unittest.TestCase):

    def test_optimize(self):

        # Load data
        sd = stock_data.StockData()
        sd.load_from_excel("/data/Factor Upload Sheet Minimal.xlsx")
        self.assertEqual(sd.get_factor_indexes(), [0,1,2,3])

        # Optimize
        fo = optimize.FactorOptimizer(
            sd, 
            st_months=6, st_weight=0.0,
            min_factors=2, max_factors=3,
            score_pairs_range=(6,10),
            random_seed=23892
            )

        fo.run(10)
        self.assertEqual(len(fo.combinations), 10)
        for combination in fo.combinations:
            self.assertGreaterEqual(len(combination['Factors']), 2)
            self.assertLessEqual(len(combination['Factors']), 3)

        fo.run(5)
        self.assertEqual(len(fo.combinations), 15)

        # Check that combinations are sorted
        for combination in fo.combinations:
            self.assertEqual(combination['Factors'], sorted(combination['Factors']))

        # Check that scores are calculated
        for combination in fo.combinations:
            self.assertFalse(np.isnan(combination['Score']))

        # Set starting point
        fo = optimize.FactorOptimizer(
            sd, 
            st_months=6, st_weight=1.0,
            min_factors=2, max_factors=3,
            score_pairs_range=(6,10),
            random_seed=82629
            )
        fo.set_starting_point({'Factors': [1,3], 'Score': 0.0 })
        fo.run(2)
        self.assertEqual(len(fo.combinations), 3)
        self.assertEqual(fo.combinations[0]['Factors'], [1,3])
