import unittest
import logging
import sys
import pandas as pd
import numpy as np
import datetime

if ('-v' in sys.argv) or ('--verbose' in sys.argv):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

from FactorCore import factor_calc, stock_data

class TestFactorCalc(unittest.TestCase):

    def test_factor_calc(self):

        # Load data
        sd = stock_data.StockData()
        sd.load_from_excel("/data/Industrials_F1_F2_Test.xlsx")
        self.assertEqual(sd.get_factor_names(), ['F1','F2'])

        # Combine factors
        combined_factors = factor_calc.combine_factors(sd.factors_df, [0,1])
        self.assertEqual(combined_factors.index.to_list(), sd.factors_df.index.to_list())

        for stock in ['Airbus SE']:
            for month in combined_factors.columns:
                f_av = combined_factors.loc[stock][month]
                f1 = sd.factors_df.loc[stock][0][month]
                f2 = sd.factors_df.loc[stock][1][month]

                if np.isnan(f1):
                    if np.isnan(f2):
                        self.assertTrue(np.isnan(f_av))
                    else:
                        self.assertEqual(f_av, f2)
                else:
                    if np.isnan(f2):
                        self.assertEqual(f_av, f1)
                    else:
                        self.assertEqual(f_av, 0.5*(f1+f2))

        # Calculate long-short pairs
        long_short_deltas = factor_calc.calc_long_short_pair_return_deltas(combined_factors, sd.returns_df, min_pairs=2, max_pairs=15)
        self.assertEqual(len(long_short_deltas.columns), 14)
        self.assertEqual(long_short_deltas.columns[0], 2)
        self.assertEqual(long_short_deltas.columns[-1], 15)
        self.assertEqual(len(long_short_deltas.index), len(combined_factors.columns))
        self.assertAlmostEqual(long_short_deltas.loc['2004-01-31'][2], 0.095872, 5)
        self.assertAlmostEqual(long_short_deltas.loc['2004-01-31'][3], 0.038362, 5)
        self.assertAlmostEqual(long_short_deltas.loc['2004-01-31'][4], 0.0, 5)
        self.assertEqual(long_short_deltas.loc['2004-01-31'][6], 0.0)

        target = np.array([ 0.06033168,  0.05071525,  0.05308702,  0.01349284,  0.03571736,
            0.01327237,  0.00290131,  0.00213906,  0.00485798,  0.00749185,
            0.00955549,  0.00355073,  0.00383021, -0.00021544])
        self.assertTrue(np.allclose(long_short_deltas.loc['2019-06-30'].values, target))

        # Metrics
        metrics = factor_calc.calc_long_short_pairs_metrics(long_short_deltas, start_date=None, end_date='2019-05-31')
        target = np.array([2.44267903, 1.98135613, 1.65498219, 1.31387991, 1.06282029,
            0.83920872, 0.75852355, 0.66092584, 0.76509141, 0.72106765,
            0.58745141, 0.49132049, 0.52487942, 0.54537571])
        self.assertTrue(np.allclose(metrics['Sum'].values, target))
        target = np.array([0.61081081, 0.57837838, 0.55675676, 0.57297297, 0.58378378,
            0.57837838, 0.55135135, 0.57297297, 0.57837838, 0.61081081,
            0.58378378, 0.56216216, 0.6       , 0.56756757])
        self.assertTrue(np.allclose(metrics['FractionPositive'].values, target))
        target = np.array([-3.255771  , -2.75507163, -2.42522893, -2.22378285, -2.1765561 ,
            -2.06177309, -2.01432834, -1.83858795, -1.67501332, -1.55319973,
            -1.58884596, -1.45073843, -1.39090587, -1.29881983])
        self.assertTrue(np.allclose(metrics['SumNegative'].values, target))
        target = [72, 78, 82, 79, 77, 78, 83, 79, 78, 72, 77, 81, 74, 80]
        self.assertEqual(metrics['CountNegative'].to_list(), target)
        target = [185, 185, 185, 185, 185, 185, 185, 185, 185, 185, 185, 185, 185, 185]
        self.assertEqual(metrics['Count'].to_list(), target)

        metrics = factor_calc.calc_long_short_pairs_metrics(long_short_deltas, start_date='2016-12-31', end_date='2019-05-31')
        target = np.array([-0.08051972, -0.1038106 , -0.16611775, -0.15880371, -0.20558809,
            -0.21874852, -0.1817987 , -0.24027093, -0.19591871, -0.1458723 ,
            -0.1455777 , -0.093803  , -0.05744669, -0.05251474])
        self.assertTrue(np.allclose(metrics['Sum'].values, target))
        target = np.array([0.46666667, 0.53333333, 0.5       , 0.46666667, 0.43333333,
            0.46666667, 0.43333333, 0.46666667, 0.43333333, 0.46666667,
            0.4       , 0.36666667, 0.5       , 0.53333333])
        self.assertTrue(np.allclose(metrics['FractionPositive'].values, target))
        target = np.array([-0.54933853, -0.49478865, -0.43663958, -0.3993307 , -0.45809061,
            -0.4673026 , -0.40532964, -0.45589821, -0.39628273, -0.34937361,
            -0.34817515, -0.30297929, -0.27155615, -0.24815644])
        self.assertTrue(np.allclose(metrics['SumNegative'].values, target))
        target = [16, 14, 15, 16, 17, 16, 17, 16, 17, 16, 18, 19, 15, 14]
        self.assertEqual(metrics['CountNegative'].to_list(), target)
        target = [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]
        self.assertEqual(metrics['Count'].to_list(), target)

    def test_combine_factors(self):

        idx = pd.MultiIndex.from_product([[0,1],['2019-01-31','2019-02-28']], names=['Factor','Date'])
        df = pd.DataFrame(index=['S1','S2'], columns=idx)
        df.loc[:,(0,'2019-01-31')] = [np.nan, 2.0]
        df.loc[:,(1,'2019-01-31')] = [1.1, 0.0]
        df.loc[:,(0,'2019-02-28')] = [1.0, 2.0]
        df.loc[:,(1,'2019-02-28')] = [3.0, 4.0]

        av = factor_calc.combine_factors(df, [0,1])
        self.assertEqual(av['2019-01-31']['S1'], 1.1)
        self.assertAlmostEqual(av['2019-01-31']['S2'], 0.5*(2.0+0.0))
        self.assertAlmostEqual(av['2019-02-28']['S1'], 0.5*(1.0+3.0))
        self.assertAlmostEqual(av['2019-02-28']['S2'], 0.5*(2.0+4.0))

        w_av = factor_calc.combine_factors_weighted(df, [0,1], [1.0, 9.0])
        self.assertEqual(w_av['2019-01-31']['S1'], 1.1)
        self.assertAlmostEqual(w_av['2019-01-31']['S2'], 0.1*2.0+0.9*0.0)
        self.assertAlmostEqual(w_av['2019-02-28']['S1'], 0.1*1.0+0.9*3.0)
        self.assertAlmostEqual(w_av['2019-02-28']['S2'], 0.1*2.0+0.9*4.0)

        w_av_equal = factor_calc.combine_factors_weighted(df, [0,1], [3.0, 3.0])
        for col in av.columns:
            self.assertEqual(list(av[col].values), list(w_av_equal[col].values))
