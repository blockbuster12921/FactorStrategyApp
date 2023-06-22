import unittest
import logging
import sys
import pandas as pd
import numpy as np
import datetime

if ('-v' in sys.argv) or ('--verbose' in sys.argv):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

from FactorCore import analysis

class TestAnalysis(unittest.TestCase):

    def test_calc_factor_weights(self):

        df = pd.DataFrame(columns=[0,1,2,3,'Score'])
        df.loc[0] = [1,0,1,1,3.0]
        df.loc[1] = [1,1,0,1,2.0]
        df.loc[2] = [1,0,0,0,2.5]
        df.loc[3] = [1,1,1,1,2.6]
        df.loc[4] = [0,0,1,1,4.0]
        df.loc[5] = [0,1,1,1,4.5]
        df.loc[6] = [0,1,1,0,3.5]
        df.loc[7] = [0,0,1,0,2.2]
        df.loc[8] = [1,0,1,0,2.8]
        df.loc[9] = [0,1,0,1,1.5]

        result = analysis.calc_factor_weights(
            df,
            factor_count=4,
            equal_weights=False,
            min_score=None,
            top_n_combinations=None,
            top_percent_combinations=None,
            exclude_top_percent_combinations=None,
            top_n_factors=None,
            min_occurrences=None,
            exclude_top_n_factors=0)
        self.assertEqual(result['Factors'], [2, 3, 1, 0])
        self.assertLess(max(abs(result['Weights']-[0.30434783, 0.26086957, 0.2173913 , 0.2173913 ])), 1e-6)
        self.assertLess(max(abs(result['Weights'] - result['Frequencies']/sum(result['Frequencies']))), 1e-6)

        result = analysis.calc_factor_weights(
            df,
            factor_count=4,
            equal_weights=True,
            min_score=None,
            top_n_combinations=None,
            top_percent_combinations=None,
            exclude_top_percent_combinations=None,
            top_n_factors=None,
            min_occurrences=None,
            exclude_top_n_factors=0)
        self.assertEqual(result['Factors'], [2, 3, 1, 0])
        self.assertListEqual(list(result['Weights']), [0.25]*4)

        result = analysis.calc_factor_weights(
            df,
            factor_count=4,
            equal_weights=False,
            min_score=3.0,
            top_n_combinations=None,
            top_percent_combinations=None,
            exclude_top_percent_combinations=None,
            top_n_factors=None,
            min_occurrences=None,
            exclude_top_n_factors=0)
        self.assertEqual(result['Factors'], [2, 3, 1, 0])
        self.assertLess(max(abs(result['Weights']-[0.4, 0.3, 0.2, 0.1])), 1e-6)
        self.assertLess(max(abs(result['Weights'] - result['Frequencies']/sum(result['Frequencies']))), 1e-6)

        result = analysis.calc_factor_weights(
            df,
            factor_count=4,
            equal_weights=False,
            min_score=None,
            top_n_combinations=4,
            top_percent_combinations=None,
            exclude_top_percent_combinations=None,
            top_n_factors=None,
            min_occurrences=None,
            exclude_top_n_factors=0)
        self.assertEqual(result['Factors'], [2, 3, 1, 0])
        self.assertLess(max(abs(result['Weights']-[0.4, 0.3, 0.2, 0.1])), 1e-6)
        self.assertLess(max(abs(result['Weights'] - result['Frequencies']/sum(result['Frequencies']))), 1e-6)

        result = analysis.calc_factor_weights(
            df,
            factor_count=4,
            equal_weights=False,
            min_score=None,
            top_n_combinations=None,
            top_percent_combinations=None,
            exclude_top_percent_combinations=None,
            top_n_factors=3,
            min_occurrences=None,
            exclude_top_n_factors=0)
        self.assertEqual(result['Factors'], [2, 3, 1, 0])
        self.assertLess(max(abs(result['Weights']-[0.30434783, 0.26086957, 0.2173913 , 0.2173913 ])), 1e-6)
        self.assertLess(max(abs(result['Weights'] - result['Frequencies']/sum(result['Frequencies']))), 1e-6)

        result = analysis.calc_factor_weights(
            df,
            factor_count=4,
            equal_weights=False,
            min_score=None,
            top_n_combinations=None,
            top_percent_combinations=None,
            exclude_top_percent_combinations=None,
            top_n_factors=2,
            min_occurrences=None,
            exclude_top_n_factors=0)
        self.assertEqual(result['Factors'], [2, 3])
        self.assertLess(max(abs(result['Weights']-[0.53846154, 0.46153846])), 1e-6)
        self.assertLess(max(abs(result['Weights'] - result['Frequencies']/sum(result['Frequencies']))), 1e-6)

        result = analysis.calc_factor_weights(
            df,
            factor_count=4,
            equal_weights=False,
            min_score=None,
            top_n_combinations=None,
            top_percent_combinations=None,
            exclude_top_percent_combinations=None,
            top_n_factors=None,
            min_occurrences=5,
            exclude_top_n_factors=0)
        self.assertEqual(result['Factors'], [2, 3, 1, 0])
        self.assertLess(max(abs(result['Weights']-[0.30434783, 0.26086957, 0.2173913 , 0.2173913 ])), 1e-6)
        self.assertLess(max(abs(result['Weights'] - result['Frequencies']/sum(result['Frequencies']))), 1e-6)

        result = analysis.calc_factor_weights(
            df,
            factor_count=4,
            equal_weights=False,
            min_score=None,
            top_n_combinations=None,
            top_percent_combinations=None,
            exclude_top_percent_combinations=None,
            top_n_factors=None,
            min_occurrences=6,
            exclude_top_n_factors=0)
        self.assertEqual(result['Factors'], [2, 3])
        self.assertLess(max(abs(result['Weights']-[0.53846154, 0.46153846])), 1e-6)
        self.assertLess(max(abs(result['Weights'] - result['Frequencies']/sum(result['Frequencies']))), 1e-6)

        result = analysis.calc_factor_weights(
            df,
            factor_count=4,
            equal_weights=False,
            min_score=None,
            top_n_combinations=None,
            top_percent_combinations=None,
            exclude_top_percent_combinations=None,
            top_n_factors=None,
            min_occurrences=None,
            exclude_top_n_factors=1)
        self.assertEqual(result['Factors'], [1, 0, 3])
        self.assertLess(max(abs(result['Weights']-[0.3125, 0.3125, 0.375])), 1e-6)
        self.assertLess(max(abs(result['Weights'] - result['Frequencies']/sum(result['Frequencies']))), 1e-6)

        result = analysis.calc_factor_weights(
            df,
            factor_count=4,
            equal_weights=False,
            min_score=None,
            top_n_combinations=None,
            top_percent_combinations=None,
            exclude_top_percent_combinations=None,
            top_n_factors=None,
            min_occurrences=None,
            exclude_top_n_factors=2)
        self.assertEqual(result['Factors'], [1, 0])
        self.assertListEqual(list(result['Weights']), [0.5]*2)
        self.assertLess(max(abs(result['Weights'] - result['Frequencies']/sum(result['Frequencies']))), 1e-6)
