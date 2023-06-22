import unittest
import logging
import sys
import pandas as pd
import datetime

if ('-v' in sys.argv) or ('--verbose' in sys.argv):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

from FactorCore import settings

class TestSettings(unittest.TestCase):

    def test_settings(self):

        pass
