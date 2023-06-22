import unittest
import logging
import sys
import pandas as pd
import numpy as np
import datetime

if ('-v' in sys.argv) or ('--verbose' in sys.argv):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

from FactorCore import score_calc

class TestScoreCalc(unittest.TestCase):

    def test_calc_score(self):

        metrics = pd.DataFrame(index=[7,8])
        metrics['Mean'] = [0.02, 0.03]
        metrics['FractionPositive'] = [0.7,0.6]
        metrics['MeanNegative'] = [-0.04, -0.02]
        score = score_calc.calc_score(metrics)
        self.assertAlmostEqual(score, 0.8)


