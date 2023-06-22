import unittest
import logging
import sys
import pandas as pd
import numpy as np

if ('-v' in sys.argv) or ('--verbose' in sys.argv):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

from FactorCore import data_io, driver

class TestDriver(unittest.TestCase):

    def test_driver(self):

        loader = data_io.DataLoader()
        loader.load_from_excel("/data/HEALTHCARE - Data Upload Minimal.xlsx")
        self.assertEqual(loader.factor_names[0], 'F1')

        f1 = loader.factors_df.loc[0]

        # Fixed
        driver_inst = driver.Driver('fixed', loader.returns_df, { 0: f1 },
                        forward_month_stocks_disabled=[],
                        in_sample_date_range=('2004-12-31', '2016-11-30'),
                        output_date_range=('2016-12-31', None),
                        factor_outlier_rejection_multiplier=2.0,
                        ranked_returns_stddev_multiplier=1.1
                    )
        driver_inst.run()

        result = driver_inst.factors_expected_returns[0]
        self.assertAlmostEqual(result.loc[58, "2018-03-31"], 0.008505581316203227)
        self.assertAlmostEqual(result.loc[42, "2017-02-28"], 0.012227011313362741)
        self.assertAlmostEqual(result.loc[65, "2017-06-30"], 0.014895606773748337)
        self.assertAlmostEqual(result.loc[51, "2017-06-30"], 0.007424138247885416)
        self.assertAlmostEqual(result.loc[57, "2018-03-31"], 0.002294892006734279)
        self.assertAlmostEqual(result.loc[46, "2017-06-30"], 0.014716052062601683)
        self.assertAlmostEqual(result.loc[38, "2019-05-31"], 0.021640159524411156)
        self.assertAlmostEqual(result.loc[52, "2019-10-31"], 0.014637477518930677)
        self.assertAlmostEqual(result.loc[41, "2019-07-31"], 0.013213055739659063)
        self.assertAlmostEqual(result.loc[56, "2019-08-31"], 0.002211398937547169)
        self.assertAlmostEqual(result.loc[63, "2016-12-31"], 0.002294892006734279)
        self.assertAlmostEqual(result.loc[55, "2019-04-30"], 0.003194798616530513)
        self.assertAlmostEqual(result.loc[50, "2018-10-31"], 0.003194798616530513)
        self.assertAlmostEqual(result.loc[23, "2019-02-28"], 0.0018595887444782027)
        self.assertAlmostEqual(result.loc[42, "2019-06-30"], 0.019089468079821773)
        self.assertAlmostEqual(result.loc[26, "2018-06-30"], 0.02754865032036545)
        self.assertAlmostEqual(result.loc[10, "2017-02-28"], 0.01716954542436715)
        self.assertAlmostEqual(result.loc[33, "2019-04-30"], 0.023996787335513827)
        self.assertAlmostEqual(result.loc[30, "2018-02-28"], 0.004474258994024991)
        self.assertTrue(np.isnan(result.loc[53, "2017-08-31"]))

        # Rolling
        driver_inst = driver.Driver('rolling', loader.returns_df, { 0: f1 },
                        forward_month_stocks_disabled=[],
                        in_sample_date_range=('2004-12-31', '2009-11-30'),
                        output_date_range=('2009-12-31', None),
                        factor_outlier_rejection_multiplier=2.0,
                        ranked_returns_stddev_multiplier=1.1
                    )
        driver_inst.run()

        result = driver_inst.factors_expected_returns[0]
        self.assertAlmostEqual(result.loc[37, "2012-03-31"], -0.0028280607730324548)
        self.assertAlmostEqual(result.loc[37, "2015-06-30"], 0.02847133728355115)
        self.assertAlmostEqual(result.loc[26, "2018-10-31"], 0.02456630612987559)
        self.assertAlmostEqual(result.loc[59, "2019-03-31"], 0.011935678973134637)
        self.assertAlmostEqual(result.loc[8, "2016-07-31"], 0.016050459619039507)
        self.assertAlmostEqual(result.loc[59, "2010-02-28"], 0.028416045615092145)
        self.assertAlmostEqual(result.loc[25, "2015-08-31"], 0.006436839547989899)
        self.assertAlmostEqual(result.loc[57, "2014-12-31"], 0.036356029670031)
        self.assertAlmostEqual(result.loc[1, "2018-09-30"], 0.013398311218507628)
        self.assertAlmostEqual(result.loc[28, "2013-09-30"], 0.0007584900121182442)
        self.assertAlmostEqual(result.loc[44, "2013-12-31"], 0.03785528934311057)
        self.assertAlmostEqual(result.loc[34, "2019-02-28"], 0.013601774340728684)
        self.assertAlmostEqual(result.loc[60, "2015-06-30"], 0.020599035157235268)
        self.assertAlmostEqual(result.loc[13, "2012-12-31"], 0.026441983284840733)
        self.assertAlmostEqual(result.loc[48, "2019-09-30"], 0.016773107955894484)
        self.assertAlmostEqual(result.loc[55, "2017-12-31"], 0.0017028684905287487)
        self.assertAlmostEqual(result.loc[58, "2010-11-30"], 0.011800159594003686)
        self.assertAlmostEqual(result.loc[14, "2013-09-30"], 0.006603840703861199)
        self.assertAlmostEqual(result.loc[34, "2013-12-31"], 0.01267029553351659)
        self.assertAlmostEqual(result.loc[27, "2016-05-31"], -0.005251609409695696)


        # Rolling with stocks disabled for forward month
        driver_inst = driver.Driver('rolling', loader.returns_df, { 0: f1 }, 
                        forward_month_stocks_disabled=[0,3],
                        in_sample_date_range=('2004-12-31', '2009-11-30'),
                        output_date_range=('2009-12-31', None),
                        factor_outlier_rejection_multiplier=2.0,
                        ranked_returns_stddev_multiplier=1.1
                    )
        driver_inst.run()

        result = driver_inst.factors_expected_returns[0]
        self.assertTrue(np.isnan(result.loc[0, "2020-01-31"]))
        self.assertTrue(np.isnan(result.loc[3, "2020-01-31"]))
        self.assertAlmostEqual(result.loc[37, "2012-03-31"], -0.0028280607730324548)
        self.assertAlmostEqual(result.loc[37, "2015-06-30"], 0.02847133728355115)
        self.assertAlmostEqual(result.loc[26, "2018-10-31"], 0.02456630612987559)
        self.assertAlmostEqual(result.loc[59, "2019-03-31"], 0.011935678973134637)
        self.assertAlmostEqual(result.loc[8, "2016-07-31"], 0.016050459619039507)
        self.assertAlmostEqual(result.loc[59, "2010-02-28"], 0.028416045615092145)
        self.assertAlmostEqual(result.loc[25, "2015-08-31"], 0.006436839547989899)
        self.assertAlmostEqual(result.loc[57, "2014-12-31"], 0.036356029670031)
        self.assertAlmostEqual(result.loc[1, "2018-09-30"], 0.013398311218507628)
        self.assertAlmostEqual(result.loc[28, "2013-09-30"], 0.0007584900121182442)
        self.assertAlmostEqual(result.loc[44, "2013-12-31"], 0.03785528934311057)
        self.assertAlmostEqual(result.loc[34, "2019-02-28"], 0.013601774340728684)
        self.assertAlmostEqual(result.loc[60, "2015-06-30"], 0.020599035157235268)
        self.assertAlmostEqual(result.loc[13, "2012-12-31"], 0.026441983284840733)
        self.assertAlmostEqual(result.loc[48, "2019-09-30"], 0.016773107955894484)
        self.assertAlmostEqual(result.loc[55, "2017-12-31"], 0.0017028684905287487)
        self.assertAlmostEqual(result.loc[58, "2010-11-30"], 0.011800159594003686)
        self.assertAlmostEqual(result.loc[14, "2013-09-30"], 0.006603840703861199)
        self.assertAlmostEqual(result.loc[34, "2013-12-31"], 0.01267029553351659)
        self.assertAlmostEqual(result.loc[27, "2016-05-31"], -0.005251609409695696)
