import pandas as pd
import numpy as np
import logging
import time
from . import database, driver, stock_selection, settings, stock_data, optimize, generate, factor_filter, factor_disable, factor_cluster
from .factor_strategy import FactorStrategy
from .utils import MeasureTime, MeasureBlockTime

logger = logging.getLogger(__name__)

class ProjectRunner():

    def __init__(self, db, project_id):
        self.db = db
        self.project_id = project_id
        self.project = None
        self.project_data_info = None
        self.settings = None
        self.target_month = None
        self.forward_month = None

        self.stages = [
            'DisableStocks','DisableFactors',
            'FactorClustering',
            #'FactorOutlierRejectionSelection','ReturnsOutlierRejectionSelection',
            'Driver','FactorFilter',
            #'TickerSelection',
            'FactorOptimize','FactorGenerate','FactorStrategies',
            ]

    def run(self):

        run_state = self.db.get_project_run_state(self.project_id)
        logger.info("Run state: {}".format(run_state))

        if run_state is not None:
            if run_state.get('RunComplete') is not None:
                # Run already marked as complete
                return

        if (run_state is None) or (len(run_state['Updates']) == 0):
            last_run_update = None
        else:
            last_run_update = run_state['Updates'][-1]

        # Initialize forward month
        self._load_project_data_info()
        self.forward_month = pd.Timestamp(self.project_data_info['Dates'][-1])

        # Initialize target month
        if last_run_update is not None:
            self.target_month = last_run_update['TargetMonth']
        else:
            self._load_project()
            if self.project['Status'] == 'Live':
                self._load_project_data_info()
                self.target_month = self.forward_month
            else:
                assert(self.project['Status'] == 'Test')

                start_date = self.project.get('OOSStartDate')
                if start_date is None:
                    self.db.update_project_run_state(self.project_id, { 'Stage': '', 'Status': 'Failed', 'TargetMonth': None, 'Detail': "Test start date not specified" })
                    return

                self.target_month = pd.Timestamp(start_date)

                # Check that target month is not >= forward month
                if self.target_month >= self.forward_month:
                    self.db.update_project_run_state(self.project_id, { 'Stage': '', 'Status': 'Failed', 'TargetMonth': None, 'Detail': "No test months to run" })
                    return

                # Check that OOS start date is <= OOS end date
                end_date = self.project.get('OOSEndDate')
                if end_date is not None:
                    end_date = pd.Timestamp(end_date)
                    if end_date < self.target_month:
                        self.db.update_project_run_state(self.project_id, { 'Stage': '', 'Status': 'Failed', 'TargetMonth': None, 'Detail': "Test start date is after end date" })
                        return

        # Check if target month is complete and move to next target month if there is one
        if last_run_update is not None:
            if (last_run_update['Stage'] == self.stages[-1]) and (last_run_update['Status'] == 'Complete'):

                # Target month is complete
                self._load_project()
                if self.project['Status'] == 'Live':
                    # Only one target month for a Live project - we are done
                    self.db.set_project_run_complete(self.project_id)
                    return

                assert(self.project['Status'] == 'Test')

                # Increment target month
                self.target_month = (pd.Timestamp(self.target_month) + pd.DateOffset(months=1)) + pd.offsets.MonthEnd(0)

                # Check against OOS end date
                end_date = self.project.get('OOSEndDate')
                if end_date is not None:
                    end_date = pd.Timestamp(end_date)
                    if self.target_month > end_date:
                        self.db.set_project_run_complete(self.project_id)
                        return

                # If target month is forward month, we are done
                if self.target_month == self.forward_month:
                    self.db.set_project_run_complete(self.project_id)
                    return

                last_run_update = None
        
        stage = None

        if last_run_update is None:
            stage = self.stages[0]
        else:
            if last_run_update['Status'] == 'Complete':
                stage_index = self.stages.index(last_run_update['Stage']) + 1
                assert(stage_index < len(self.stages))
                stage = self.stages[stage_index]
            else:
                stage = last_run_update['Stage']

        self._run_stage(stage)

    def _run_stage(self, stage):

        logger.info("Running stage '{}'".format(stage))

        self.stage = stage

        try:
            if stage == 'DisableStocks':
                self._run_disable_stocks()
            if stage == 'DisableFactors':
                self._run_disable_factors()
            if stage == 'FactorClustering':
                self._run_factor_clustering()
            if stage == 'TickerSelection':
                self._run_ticker_selection()
            if stage == 'FactorOutlierRejectionSelection':
                self._run_factor_outlier_param_select()
            if stage == 'ReturnsOutlierRejectionSelection':
                self._run_returns_outlier_param_select()
            if stage == 'Driver':
                self._run_driver()
            if stage == 'FactorFilter':
                self._run_factor_filter()
            if stage == 'FactorOptimize':
                self._run_factor_optimize()
            if stage == 'FactorGenerate':
                self._run_factor_generate()
            if stage == 'FactorStrategies':
                self._run_factor_strategies()
        except Exception as e:
            self.db.update_project_run_state(
                self.project_id, 
                { 'Stage': self.stage, 'Status': 'Failed', 'TargetMonth': self.target_month, 'Detail': "Unexpected error: {}".format(str(e)) }
                )

    def _load_project(self):
        if self.project is None:
            self.project = self.db.get_project(self.project_id)

    def _load_project_data_info(self):
        if self.project_data_info is None:
            self.project_data_info = self.db.get_project_data_info(self.project_id)

    def _init_settings(self):
        if self.settings is None:
            self.settings = self.db.get_project_settings(self.project_id)
            if self.settings is None:
                self.settings = {}
            self.settings = settings.overlay_default_project_settings(self.settings)

    def _perform_returns_outlier_rejection(self, returns_df):
        
        # Rejection of entire month based on extreme average return for that month
        r_average = returns_df.median(axis=1)

        lo = self.settings['AverageReturnsOutlierRangeStart'] / 100.0
        hi = self.settings['AverageReturnsOutlierRangeEnd'] / 100.0

        if lo >= hi:
            raise ValueError('Average Returns Outlier Range is invalid')

        returns_df.loc[(r_average < lo) | (r_average > hi)] *= 0.0

        # Rejection of individual returns based on outlier detection for each month
        monthly_returns_outlier_method = self.settings['ReturnsMonthlyOutlierRejectionMethod']
        if monthly_returns_outlier_method != "none":

            x = returns_df.T
            if monthly_returns_outlier_method == "standard":
                z = (x-x.mean())/x.std()
            elif monthly_returns_outlier_method == "robust":
                z = 0.6745*(x - x.median())/((x - x.median()).abs().median())
            else:
                raise ValueError('Invalid ReturnsMonthlyOutlierRejectionMethod')

            returns_df = returns_df.mask(z.T.abs() > self.settings['ReturnsMonthlyOutlierRejectionCutoff'])

        return returns_df
    
    def _run_disable_stocks(self):

        logger.info("Running disable stocks")

        self._load_project_data_info()
        self._init_settings()

        stocks_enabled = list(range(0, len(self.project_data_info['Stocks'])))

        stocks_disabled = stock_selection.get_indexes_for_disabled_stocks(
            self.project_data_info['Stocks'], self.db.get_project_stocks_disabled(self.project_id))

        if len(stocks_disabled) > 0:
            stocks_enabled = list(set(stocks_enabled) - set(stocks_disabled))

        if len(stocks_disabled) == 0:
            detail = "No stocks disabled by user"
        else:
            detail = "{} stocks disabled by user: {}".format(len(stocks_disabled), ", ".join([self.project_data_info['Stocks'][stock]['Name'] for stock in stocks_disabled]))
        self.db.update_project_run_state(self.project_id, { 'Stage': self.stage, 'Status': 'InProgress', 'TargetMonth': self.target_month, 'Detail': detail })

        # Load data
        market_cap_factor_index = self.project_data_info['MarketCapFactorIndex']
        data = self.db.get_project_data(self.project_id, self.project_data_info, [market_cap_factor_index])

        # Disable stocks with no returns
        returns_count = data['Returns'].loc[:self.target_month].count(axis=0)
        empty_stocks = returns_count.loc[returns_count == 0].index.tolist()

        if len(empty_stocks) > 0:
            stocks_enabled = list(set(stocks_enabled) - set(empty_stocks))
            detail = "Disabled {} stocks with no returns data: {}".format(len(empty_stocks), ", ".join([self.project_data_info['Stocks'][stock]['Name'] for stock in empty_stocks]))
            self.db.update_project_run_state(self.project_id, { 'Stage': self.stage, 'Status': 'InProgress', 'TargetMonth': self.target_month, 'Detail': detail })

        # Filter stocks by target month market cap

        try:
            market_cap_df = data['Factors'][market_cap_factor_index]
        except Exception as e:
            raise ValueError("Failed to get Market Cap from Factor data")

        market_cap_filter = self.settings['MarketCapFilterValue']
        try:
            market_cap_filter_date = pd.Timestamp(self.settings['MarketCapFilterDate'])
            reference_market_cap = market_cap_df.loc[market_cap_filter_date].loc[stocks_enabled]
        except Exception as e:
            self.db.update_project_run_state(self.project_id, { 'Stage': self.stage, 'Status': 'Failed', 'TargetMonth': self.target_month,
                'Detail': 'Invalid Market Cap filter date' })
            return

        retained_count = len(reference_market_cap.loc[reference_market_cap >= market_cap_filter])
        self.db.update_project_run_state(self.project_id, { 'Stage': self.stage, 'Status': 'InProgress', 'TargetMonth': self.target_month,
            'Detail': "{} enabled stocks above Market Cap filter {}".format(retained_count, market_cap_filter) })

        target_month_market_cap = market_cap_df.loc[self.target_month].loc[stocks_enabled]            
        stocks_enabled_after_market_cap_filter = target_month_market_cap.sort_values().tail(retained_count).index.tolist()

        stocks_removed_by_market_cap_filter = list(set(stocks_enabled).difference(set(stocks_enabled_after_market_cap_filter)))
        stocks_removed_by_market_cap_filter_names = [self.project_data_info['Stocks'][stock]['Name'] for stock in stocks_removed_by_market_cap_filter]
        if len(stocks_removed_by_market_cap_filter_names) == 0:
            detail = "No stocks filtered by Market Cap"
        else:
            detail = "{} stocks filtered by Market Cap: {}".format(len(stocks_removed_by_market_cap_filter_names), ", ".join(stocks_removed_by_market_cap_filter_names))
        self.db.update_project_run_state(self.project_id, { 'Stage': self.stage, 'Status': 'InProgress', 'TargetMonth': self.target_month, 'Detail': detail })

        # Initialise and save stock selection state
        state = stock_selection.StockSelector.create_state(stocks_enabled, self.settings['TickerSelectionMinStocks'], stocks_removed_by_market_cap_filter)
        self.db.update_project_stock_selection_state(self.project_id, self.target_month, state)

        self.db.update_project_run_state(self.project_id, { 'Stage': self.stage, 'Status': 'Complete', 'TargetMonth': self.target_month, 'Detail': '' })

    def _run_disable_factors(self):
    
        logger.info("Running disable factors")

        self._load_project_data_info()
        self._init_settings()

        # Get enabled stocks
        stock_selection_state = self.db.get_project_stock_selection_state(self.project_id, self.target_month)
        stocks_enabled = stock_selection.StockSelector.get_enabled_stocks_from_state(stock_selection_state, apply_market_cap_filter=(self.stage in self.settings['MarketCapFilterStages']))

        # Get enabled factors
        factors_enabled = list(range(0, len(self.project_data_info['Factors'])))
        factors_disabled = self.db.get_project_factors_disabled(self.project_id)

        if (factors_disabled is not None) and (len(factors_disabled) > 0):
            factors_disabled = [self.project_data_info['Factors'].index(f) for f in factors_disabled]
            factors_enabled = list(set(factors_enabled) - set(factors_disabled))
            detail = "{} factors disabled by user: {}".format(
                len(factors_disabled),
                ",".join([self.project_data_info['Factors'][f] for f in factors_disabled])
            )
        else:
            detail = "No factors disabled by user"
        self.db.update_project_run_state(self.project_id, { 'Stage': self.stage, 'Status': 'InProgress', 'TargetMonth': self.target_month, 'Detail': detail })

        # Load factor data
        data = self.db.get_project_data(self.project_id, self.project_data_info, factors_enabled)

        # Determine date range
        date_range = ((self.target_month - pd.DateOffset(months=self.settings['FactorSelectionPeriodDuration'])) + pd.offsets.MonthEnd(0), self.target_month)

        # Run factor disabling
        state = factor_disable.FactorDisabler.create_state(factors_enabled)

        disabler = factor_disable.FactorDisabler(
            factors_data=data['Factors'],
            date_range=date_range,
            required_data_fraction=float(self.settings['FactorDataCompletenessPercentage'])/100.0,
            stocks_enabled=stocks_enabled,
            state=state,
        )

        disabler.run()

        # Save state
        self.db.update_project_factor_disabled_state(self.project_id, self.target_month, state)

        # Update run history
        incomplete_factors = [item['Factor'] for item in state['FactorDataComplete'] if not item['Enabled']]
        if len(incomplete_factors) == 0:
            detail = "No factors disabled because of incomplete data"
        else:
            detail = "{} factors disabled because of incomplete data: {}".format(
                len(incomplete_factors),
                ",".join([self.project_data_info['Factors'][f] for f in incomplete_factors])
            )
        self.db.update_project_run_state(self.project_id, { 'Stage': self.stage, 'Status': 'InProgress', 'TargetMonth': self.target_month, 'Detail': detail })

        factors_enabled = factor_disable.FactorDisabler.get_enabled_factors_from_state(state)
        if len(factors_enabled) == 0:
            self.db.update_project_run_state(self.project_id, { 'Stage': self.stage, 'Status': 'Failed', 'TargetMonth': self.target_month, 'Detail': 'No factors enabled' })
            return

        self.db.update_project_run_state(self.project_id, { 'Stage': self.stage, 'Status': 'Complete', 'TargetMonth': self.target_month, 'Detail': '' })

    def _run_factor_clustering(self):

        logger.info("Running factor clustering")

        self._load_project_data_info()
        self._init_settings()

        # Load data
        factors_enabled = factor_disable.FactorDisabler.get_enabled_factors_from_state(
            self.db.get_project_factor_disabled_state(self.project_id, self.target_month))

        data = self.db.get_project_data(self.project_id, self.project_data_info, factors_enabled)

        # Perform outlier rejection on average returns
        data['Returns'] = self._perform_returns_outlier_rejection(data['Returns'])

        # Run clustering
        end_date = (self.target_month - pd.DateOffset(months=1)) + pd.offsets.MonthEnd(0)
        start_date = (end_date - pd.DateOffset(months=self.settings['FactorClusteringDurationMonths']-1)) + pd.offsets.MonthEnd(0)
        start_date = max(start_date, pd.Timestamp(self.settings['FactorClusteringStartDate']))

        stock_selection_state = self.db.get_project_stock_selection_state(self.project_id, self.target_month)
        stocks_enabled = stock_selection.StockSelector.get_enabled_stocks_from_state(stock_selection_state, apply_market_cap_filter=(self.stage in self.settings['MarketCapFilterStages']))

        fcg = factor_cluster.FactorClusterGenerator(
            returns_df=data['Returns'],
            factor_dfs=data['Factors'],
            stocks_enabled=stocks_enabled,
            start_date=start_date,
            end_date=end_date,
            distance_threshold_multiplier=self.settings['FactorClusteringDistanceThresholdMultiplier'],
        )

        clusters = fcg.run()

        # Save clusters
        self.db.set_project_factor_clusters(self.project_id, self.target_month, clusters)

        # Update run state
        self.db.update_project_run_state(
            self.project_id, 
            { 'Stage': self.stage, 'Status': 'Complete', 'TargetMonth': self.target_month, 
              'Detail': "{} clusters generated for {} factors".format(len(clusters['Cluster'].unique()), len(clusters)) }
            )

    def _load_expected_returns(self, apply_market_cap_filter=True):

        # Load factor expected returns
        factor_dfs = self.db.get_project_factor_expected_returns(self.project_id, self.target_month)

        # Remove last column in factor expected returns - this will correspond to the target month
        for key, df in factor_dfs.items():
            factor_dfs[key] = df[df.columns[:-1]]

        # Get enabled stocks and apply to factors
        stock_selection_state = self.db.get_project_stock_selection_state(self.project_id, self.target_month)
        stocks_enabled = stock_selection.StockSelector.get_enabled_stocks_from_state(stock_selection_state, apply_market_cap_filter=(self.stage in self.settings['MarketCapFilterStages']))
        for key, df in factor_dfs.items():
            factor_dfs[key] = df.loc[stocks_enabled]

        # Load returns
        returns_df = self.db.get_project_data(self.project_id, self.project_data_info, factor_indexes=[])['Returns']

        # Perform outlier rejection on average returns
        returns_df = self._perform_returns_outlier_rejection(returns_df)

        # Reduce returns to required months and stocks
        returns_df = returns_df.T
        factor_df = list(factor_dfs.values())[0]
        returns_df = returns_df.loc[factor_df.index]
        returns_df = returns_df[factor_df.columns]

        return returns_df, factor_dfs

    def _run_factor_outlier_param_select(self):

        logger.info("Running Factor Outlier Rejection Parameter Selection")

        self._init_settings()

        # Load state
        state = self.db.get_project_driver_param_selection_state(self.project_id, self.target_month, 'FactorOutlierRejection')

        # Initialise param values if necessary
        param_values = None
        if state is None:

            start = float(self.settings['FactorOutlierRejectionRangeStart'])
            end = float(self.settings['FactorOutlierRejectionRangeEnd'])
            step = float(self.settings['FactorOutlierRejectionRangeStep'])
            default = float(self.settings['FactorOutlierRejection'])

            if end < start:
                self.db.update_project_run_state(self.project_id, { 'Stage': 'FactorOutlierRejectionSelection', 'Status': 'Failed', 'TargetMonth': self.target_month,
                    'Detail': 'Range start is above end' })
                return

            param_values = list(np.arange(start, end, step)) + [end]
            if default not in param_values:
                param_values = sorted(param_values + [default])

            if len(param_values) == 1:
                self.db.update_project_run_state(self.project_id, { 'Stage': 'FactorOutlierRejectionSelection', 'Status': 'Complete', 'TargetMonth': self.target_month,
                    'Detail': 'Not required' })
                return

        # Get enabled stocks & factors
        self._load_project_data_info()

        stock_selection_state = self.db.get_project_stock_selection_state(self.project_id, self.target_month)
        stocks_enabled = stock_selection.StockSelector.get_enabled_stocks_from_state(stock_selection_state, apply_market_cap_filter=(self.stage in self.settings['MarketCapFilterStages']))

        forward_month_stocks_disabled = stock_selection.get_indexes_for_disabled_stocks(
            self.project_data_info['Stocks'], self.db.get_project_stocks_forward_month_disabled(self.project_id))
        forward_month_stocks_disabled = list(set(forward_month_stocks_disabled).intersection(set(stocks_enabled)))

        factors_enabled = factor_disable.FactorDisabler.get_enabled_factors_from_state(
            self.db.get_project_factor_disabled_state(self.project_id, self.target_month))

        # Load data
        data = self.db.get_project_data(self.project_id, self.project_data_info, factors_enabled)

        # Perform outlier rejection on average returns
        data['Returns'] = self._perform_returns_outlier_rejection(data['Returns'])

        # Get settings
        driver_method = self.settings['DriverMethod']

        in_sample_date_range = (pd.Timestamp(self.settings['DriverInSampleStartDate']), pd.Timestamp(self.settings['DriverInSampleEndDate']))

        output_duration = self.settings['TickerSelectionPeriodDuration']
        output_date_range = (
            (self.target_month - pd.DateOffset(months=output_duration)) + pd.offsets.MonthEnd(0), 
            (self.target_month - pd.DateOffset(months=1)) + pd.offsets.MonthEnd(0), 
            )

        if output_date_range[0] <= in_sample_date_range[1]:
            output_date_range = (
                (in_sample_date_range[1] + pd.DateOffset(months=1)) + pd.offsets.MonthEnd(0),
                output_date_range[1]
                )

        if output_date_range[1] <= output_date_range[0]:
            detail = "Invalid time period {:%b %Y} - {:%b %Y}".format(output_date_range[0], output_date_range[1])
            self.db.update_project_run_state(self.project_id, 
                { 'Stage': 'FactorOutlierRejectionSelection', 'Status': 'Failed', 'TargetMonth': self.target_month, 'Detail': detail })
            return

        factor_outlier_rejection_multiplier = self.settings['FactorOutlierRejection']

        ranked_returns_stddev_multiplier = self.settings['DriverReturnsOutlierRejectionDefault']
        param_selection_state = self.db.get_project_driver_param_selection_state(self.project_id, self.target_month, 'ReturnsOutlierRejection')
        if (param_selection_state is not None) and param_selection_state['Complete']:
            best_trial = param_selection_state['Trials'][0]
            for trial in param_selection_state['Trials'][1:]:
                if trial['Score'] > best_trial['Score']:
                    best_trial = trial
            ranked_returns_stddev_multiplier = best_trial['ParamValue']

        pairs_range = (
            self.settings['LongShortPairsTarget']-self.settings['LongShortPairsDelta'], 
            self.settings['LongShortPairsTarget']+self.settings['LongShortPairsDelta']
            )

        driver_calc_settings = {
            'DriverReturnRankCorrelationOrder': self.settings['DriverReturnRankCorrelationOrder'],
            'DriverReturnRankCorrelationSplicePercent': self.settings['DriverReturnRankCorrelationSplicePercent'],
            'DriverReturnRankCorrelationMinDataMonths': self.settings['DriverReturnRankCorrelationMinDataMonths'],
            'DriverReturnRankTTestPercentage': self.settings['DriverReturnRankTTestPercentage'],
            'DriverBinnedReturnRankCorrelationBinPercentage': self.settings['DriverBinnedReturnRankCorrelationBinPercentage'],
            'DriverFactorWeightScale': self.settings['DriverFactorWeightScale'],
            'DriverCorrelationMinMonths': self.settings['DriverCorrelationMinMonths'],
        }

        # Run selection
        returns_data = data['Returns'].copy()[stocks_enabled]
        factors_data = { index: data['Factors'][index].copy()[stocks_enabled] for index in factors_enabled }

        param_selector = driver.DriverParamSelector(
            returns_data,
            factors_data,
            driver_method,
            forward_month_stocks_disabled,
            in_sample_date_range,
            output_date_range,
            factor_outlier_rejection_multiplier,
            ranked_returns_stddev_multiplier,
            correlation_average_method=self.settings['DriverCorrelationAverageMethod'],
            correlation_lookback_months=int(self.settings['DriverCorrelationAverageMonths']),
            correlation_basis=self.settings['DriverCorrelationBasis'],
            calculation_settings=driver_calc_settings,
            score_pairs_range=pairs_range,
            param='FactorOutlierRejection',
            state=state,
            param_values=param_values,
        )

        start_time = time.perf_counter()
        for i in range(0, 100):

            param_selector.run()

            # Save selection state
            self.db.update_project_driver_param_selection_state(self.project_id, self.target_month, 'FactorOutlierRejection', param_selector.state)

            # Update run state
            last_trial = param_selector.state['Trials'][0]
            for trial in param_selector.state['Trials'][1:]:
                if trial['Score'] is None:
                    break
                last_trial = trial
            detail = "Score is {:.3f} for parameter value {:.3g}".format(last_trial['Score'], last_trial['ParamValue'])
            self.db.update_project_run_state(self.project_id, { 'Stage': 'FactorOutlierRejectionSelection', 'Status': 'InProgress', 'TargetMonth': self.target_month, 'Detail': detail })

            # Check for completion
            if param_selector.state['Complete']:
                best_trial = param_selector.state['Trials'][0]
                for trial in param_selector.state['Trials'][1:]:
                    if trial['Score'] > best_trial['Score']:
                        best_trial = trial
                detail = "Using parameter value {:.3g} with highest score {:.3f}".format(best_trial['ParamValue'], best_trial['Score'])
                self.db.update_project_run_state(self.project_id, { 'Stage': 'FactorOutlierRejectionSelection', 'Status': 'Complete', 'TargetMonth': self.target_month, 'Detail': detail })
                return

            elapsed_time = time.perf_counter() - start_time
            if elapsed_time > 30.0:
                break


    def _run_returns_outlier_param_select(self):

        logger.info("Running Returns Outlier Rejection Parameter Selection")

        self._init_settings()

        # Load state
        state = self.db.get_project_driver_param_selection_state(self.project_id, self.target_month, 'ReturnsOutlierRejection')

        # Initialise param values if necessary
        param_values = None
        if state is None:

            start = float(self.settings['DriverReturnsOutlierRejectionRangeStart'])
            end = float(self.settings['DriverReturnsOutlierRejectionRangeEnd'])
            step = float(self.settings['DriverReturnsOutlierRejectionRangeStep'])
            default = float(self.settings['DriverReturnsOutlierRejectionDefault'])

            if end < start:
                self.db.update_project_run_state(self.project_id, { 'Stage': 'ReturnsOutlierRejectionSelection', 'Status': 'Failed', 'TargetMonth': self.target_month,
                    'Detail': 'Range start is above end' })
                return

            param_values = list(np.arange(start, end, step)) + [end]
            if default not in param_values:
                param_values = sorted(param_values + [default])

            if len(param_values) == 1:
                self.db.update_project_run_state(self.project_id, { 'Stage': 'ReturnsOutlierRejectionSelection', 'Status': 'Complete', 'TargetMonth': self.target_month,
                    'Detail': 'Not required' })
                return

        # Get enabled stocks & factors
        self._load_project_data_info()

        stock_selection_state = self.db.get_project_stock_selection_state(self.project_id, self.target_month)
        stocks_enabled = stock_selection.StockSelector.get_enabled_stocks_from_state(stock_selection_state, apply_market_cap_filter=(self.stage in self.settings['MarketCapFilterStages']))

        forward_month_stocks_disabled = stock_selection.get_indexes_for_disabled_stocks(
            self.project_data_info['Stocks'], self.db.get_project_stocks_forward_month_disabled(self.project_id))
        forward_month_stocks_disabled = list(set(forward_month_stocks_disabled).intersection(set(stocks_enabled)))

        factors_enabled = factor_disable.FactorDisabler.get_enabled_factors_from_state(
            self.db.get_project_factor_disabled_state(self.project_id, self.target_month))

        # Load data
        data = self.db.get_project_data(self.project_id, self.project_data_info, factors_enabled)

        # Perform outlier rejection on average returns
        data['Returns'] = self._perform_returns_outlier_rejection(data['Returns'])

        # Get settings
        driver_method = self.settings['DriverMethod']

        in_sample_date_range = (pd.Timestamp(self.settings['DriverInSampleStartDate']), pd.Timestamp(self.settings['DriverInSampleEndDate']))

        output_duration = self.settings['TickerSelectionPeriodDuration']
        output_date_range = (
            (self.target_month - pd.DateOffset(months=output_duration)) + pd.offsets.MonthEnd(0), 
            (self.target_month - pd.DateOffset(months=1)) + pd.offsets.MonthEnd(0), 
            )

        if output_date_range[0] <= in_sample_date_range[1]:
            output_date_range = (
                (in_sample_date_range[1] + pd.DateOffset(months=1)) + pd.offsets.MonthEnd(0),
                output_date_range[1]
                )

        if output_date_range[1] <= output_date_range[0]:
            detail = "Invalid time period {:%b %Y} - {:%b %Y}".format(output_date_range[0], output_date_range[1])
            self.db.update_project_run_state(self.project_id, 
                { 'Stage': 'ReturnsOutlierRejectionSelection', 'Status': 'Failed', 'TargetMonth': self.target_month, 'Detail': detail })
            return

        factor_outlier_rejection_multiplier = self.settings['FactorOutlierRejection']
        param_selection_state = self.db.get_project_driver_param_selection_state(self.project_id, self.target_month, 'FactorOutlierRejection')
        if (param_selection_state is not None) and param_selection_state['Complete']:
            best_trial = param_selection_state['Trials'][0]
            for trial in param_selection_state['Trials'][1:]:
                if trial['Score'] > best_trial['Score']:
                    best_trial = trial
            factor_outlier_rejection_multiplier = best_trial['ParamValue']

        ranked_returns_stddev_multiplier = self.settings['DriverReturnsOutlierRejectionDefault']

        pairs_range = (
            self.settings['LongShortPairsTarget']-self.settings['LongShortPairsDelta'], 
            self.settings['LongShortPairsTarget']+self.settings['LongShortPairsDelta']
            )

        driver_calc_settings = {
            'DriverReturnRankCorrelationOrder': self.settings['DriverReturnRankCorrelationOrder'],
            'DriverReturnRankCorrelationSplicePercent': self.settings['DriverReturnRankCorrelationSplicePercent'],
            'DriverReturnRankCorrelationMinDataMonths': self.settings['DriverReturnRankCorrelationMinDataMonths'],
            'DriverReturnRankTTestPercentage': self.settings['DriverReturnRankTTestPercentage'],
            'DriverBinnedReturnRankCorrelationBinPercentage': self.settings['DriverBinnedReturnRankCorrelationBinPercentage'],
            'DriverFactorWeightScale': self.settings['DriverFactorWeightScale'],
            'DriverCorrelationMinMonths': self.settings['DriverCorrelationMinMonths'],
        }

        # Run selection
        returns_data = data['Returns'].copy()[stocks_enabled]
        factors_data = { index: data['Factors'][index].copy()[stocks_enabled] for index in factors_enabled }

        param_selector = driver.DriverParamSelector(
            returns_data,
            factors_data,
            driver_method,
            forward_month_stocks_disabled,
            in_sample_date_range,
            output_date_range,
            factor_outlier_rejection_multiplier,
            ranked_returns_stddev_multiplier,
            correlation_average_method=self.settings['DriverCorrelationAverageMethod'],
            correlation_lookback_months=int(self.settings['DriverCorrelationAverageMonths']),
            correlation_basis=self.settings['DriverCorrelationBasis'],
            calculation_settings=driver_calc_settings,
            score_pairs_range=pairs_range,
            param='ReturnsOutlierRejection',
            state=state,
            param_values=param_values,
        )

        start_time = time.perf_counter()
        for i in range(0, 100):

            param_selector.run()

            # Save selection state
            self.db.update_project_driver_param_selection_state(self.project_id, self.target_month, 'ReturnsOutlierRejection', param_selector.state)

            # Update run state
            last_trial = param_selector.state['Trials'][0]
            for trial in param_selector.state['Trials'][1:]:
                if trial['Score'] is None:
                    break
                last_trial = trial
            detail = "Score is {:.3f} for parameter value {:.3g}".format(last_trial['Score'], last_trial['ParamValue'])
            self.db.update_project_run_state(self.project_id, { 'Stage': 'ReturnsOutlierRejectionSelection', 'Status': 'InProgress', 'TargetMonth': self.target_month, 'Detail': detail })

            # Check for completion
            if param_selector.state['Complete']:
                best_trial = param_selector.state['Trials'][0]
                for trial in param_selector.state['Trials'][1:]:
                    if trial['Score'] > best_trial['Score']:
                        best_trial = trial
                detail = "Using parameter value {:.3g} with highest score {:.3f}".format(best_trial['ParamValue'], best_trial['Score'])
                self.db.update_project_run_state(self.project_id, { 'Stage': 'ReturnsOutlierRejectionSelection', 'Status': 'Complete', 'TargetMonth': self.target_month, 'Detail': detail })
                return

            elapsed_time = time.perf_counter() - start_time
            if elapsed_time > 30.0:
                break


    def _run_driver(self):

        logger.info("Running Driver")

        self._load_project_data_info()
        self._init_settings()

        # Get enabled stocks & factors
        stock_selection_state = self.db.get_project_stock_selection_state(self.project_id, self.target_month)
        stocks_enabled = stock_selection.StockSelector.get_enabled_stocks_from_state(stock_selection_state, apply_market_cap_filter=(self.stage in self.settings['MarketCapFilterStages']))

        forward_month_stocks_disabled = stock_selection.get_indexes_for_disabled_stocks(
            self.project_data_info['Stocks'], self.db.get_project_stocks_forward_month_disabled(self.project_id))
        forward_month_stocks_disabled = list(set(forward_month_stocks_disabled).intersection(set(stocks_enabled)))

        factors_enabled = factor_disable.FactorDisabler.get_enabled_factors_from_state(
            self.db.get_project_factor_disabled_state(self.project_id, self.target_month))

        # Load data
        data = self.db.get_project_data(self.project_id, self.project_data_info, factors_enabled)

        # Perform outlier rejection on average returns
        data['Returns'] = self._perform_returns_outlier_rejection(data['Returns'])

        # Get driver settings
        driver_method = self.settings['DriverMethod']

        in_sample_date_range = (pd.Timestamp(self.settings['DriverInSampleStartDate']), pd.Timestamp(self.settings['DriverInSampleEndDate']))

        output_duration = self.settings['FactorSelectionPeriodDuration']
        output_date_range = ((self.target_month - pd.DateOffset(months=output_duration)) + pd.offsets.MonthEnd(0), self.target_month)

        if output_date_range[0] <= in_sample_date_range[1]:
            detail = "Driver In-Sample Period {:%b %Y} - {:%b %Y} overlaps with Factor Selection Period {:%b %Y} - {:%b %Y}".format(
                in_sample_date_range[0], in_sample_date_range[1], output_date_range[0], output_date_range[1])
            self.db.update_project_run_state(self.project_id, 
                { 'Stage': 'Driver', 'Status': 'Failed', 'TargetMonth': self.target_month, 'Detail': detail })
            return

        factor_outlier_rejection_multiplier = self.settings['FactorOutlierRejection']
        param_selection_state = self.db.get_project_driver_param_selection_state(self.project_id, self.target_month, 'FactorOutlierRejection')
        if (param_selection_state is not None) and param_selection_state['Complete']:
            best_trial = param_selection_state['Trials'][0]
            for trial in param_selection_state['Trials'][1:]:
                if trial['Score'] > best_trial['Score']:
                    best_trial = trial
            factor_outlier_rejection_multiplier = best_trial['ParamValue']

        ranked_returns_stddev_multiplier = self.settings['DriverReturnsOutlierRejectionDefault']
        param_selection_state = self.db.get_project_driver_param_selection_state(self.project_id, self.target_month, 'ReturnsOutlierRejection')
        if (param_selection_state is not None) and param_selection_state['Complete']:
            best_trial = param_selection_state['Trials'][0]
            for trial in param_selection_state['Trials'][1:]:
                if trial['Score'] > best_trial['Score']:
                    best_trial = trial
            ranked_returns_stddev_multiplier = best_trial['ParamValue']

        driver_calc_settings = {
            'DriverReturnRankCorrelationOrder': self.settings['DriverReturnRankCorrelationOrder'],
            'DriverReturnRankCorrelationSplicePercent': self.settings['DriverReturnRankCorrelationSplicePercent'],
            'DriverReturnRankCorrelationMinDataMonths': self.settings['DriverReturnRankCorrelationMinDataMonths'],
            'DriverReturnRankTTestPercentage': self.settings['DriverReturnRankTTestPercentage'],
            'DriverBinnedReturnRankCorrelationBinPercentage': self.settings['DriverBinnedReturnRankCorrelationBinPercentage'],
            'DriverFactorWeightScale': self.settings['DriverFactorWeightScale'],
            'DriverCorrelationMinMonths': self.settings['DriverCorrelationMinMonths'],
        }

        # Run driver
        returns_data = data['Returns'].copy()[stocks_enabled]
        factors_data = { index: data['Factors'][index].copy()[stocks_enabled] for index in factors_enabled }

        driver_instance = driver.Driver(
            method=driver_method, 
            returns_data=returns_data,
            factors_data=factors_data,
            forward_month_stocks_disabled=forward_month_stocks_disabled,
            in_sample_date_range=in_sample_date_range,
            output_date_range=output_date_range,
            factor_outlier_rejection_multiplier=factor_outlier_rejection_multiplier,
            ranked_returns_stddev_multiplier=ranked_returns_stddev_multiplier,
            correlation_average_method=self.settings['DriverCorrelationAverageMethod'],
            correlation_lookback_months=int(self.settings['DriverCorrelationAverageMonths']),
            correlation_basis=self.settings['DriverCorrelationBasis'],
            calculation_settings=driver_calc_settings,
            )

        driver_instance.run()

        # Store factor expected returns and driver weights
        expected_returns = driver_instance.factors_expected_returns
        if len(driver_instance.factor_weights) > 0:
            if self.settings['DriverTopFactors'] < len(expected_returns):
                top_factors = set(driver_instance.factor_weights.abs().nlargest(self.settings['DriverTopFactors']).index.tolist())
                expected_returns = { f: expected_returns[f] for f in top_factors }

            self.db.set_project_driver_factor_weights(self.project_id, self.target_month, driver_instance.factor_weights)

        self.db.set_project_factor_expected_returns(self.project_id, self.target_month, expected_returns)

        # Update run state
        self.db.update_project_run_state(self.project_id, { 'Stage': 'Driver', 'Status': 'Complete', 'TargetMonth': self.target_month })


    def _run_factor_filter(self):

        logger.info("Running factor filter")

        self._init_settings()

        if self.settings['DriverMethod'] == 'variable_direction':
            self.db.update_project_run_state(self.project_id, { 'Stage': self.stage, 'Status': 'Complete', 'TargetMonth': self.target_month, 
                'Detail': "No Factor Filter for Variable Direction Driver method" })
            return

        self._load_project_data_info()

        # Load returns and factor expected returns
        returns_df, factor_dfs = self._load_expected_returns()

        # Load clusters
        cluster_df = self.db.get_project_factor_clusters(self.project_id, self.target_month)

        # Filter clusters to factors with expected returns
        cluster_df = cluster_df.loc[factor_dfs.keys()]

        # Load or create state
        state = self.db.get_project_factor_filter_state(self.project_id, self.target_month)
        if state is None:
            state = factor_filter.FactorFilter.create_state(cluster_df, int(self.settings['FactorFilterMinFactorCount']))
            self.db.update_project_factor_filter_state(self.project_id, self.target_month, state)

            if state['Complete']:
                self.db.update_project_run_state(self.project_id, { 'Stage': self.stage, 'Status': 'Complete', 'TargetMonth': self.target_month, 
                    'Detail': "All factor clusters retained" })
                return

        # Create filter
        ff = factor_filter.FactorFilter(
            returns_df,
            factor_dfs,
            cluster_df,
            st_duration=int(self.settings['FactorSelectionSTPeriodDuration']),
            st_weight=float(self.settings['FactorSelectionSTPeriodWeight']),
            objective=self.settings['FactorSelectionObjective'],
            score_pairs_range=(int(self.settings['FactorSelectionLongShortPairsStart']), int(self.settings['FactorSelectionLongShortPairsEnd'])),
            removal_fraction=float(self.settings['FactorFilterRemovalPercentage'])/100.0,
            min_cluster_count=int(self.settings['FactorFilterMinFactorCount']),
            state=state
            )

        # Run filter
        start_time = time.perf_counter()
        while True:

            ff.run()

            if ff.state['Complete']:
                break

            elapsed_time = time.perf_counter() - start_time
            if elapsed_time > 30.0:
                break

        # Update state
        self.db.update_project_factor_filter_state(self.project_id, self.target_month, ff.state)

        # Update run state
        self.db.update_project_run_state(self.project_id, 
                { 'Stage': self.stage, 'Status': 'InProgress', 'TargetMonth': self.target_month, 
                'Detail': "Best after {} stages = {:.3f}".format(len(ff.state['Stages']), ff.state['Stages'][ff.state['BestStage']]['BaselineScore']) })

        if ff.state['Complete']:
            #retained_factors = [ self.project_data_info['Factors'][trial['Factor']] for trial in ff.state['Stages'][ff.state['BestStage']]['Trials'] ]
            retained_clusters = [ trial['Cluster'] for trial in ff.state['Stages'][ff.state['BestStage']]['Trials'] ]
            self.db.update_project_run_state(self.project_id, 
                { 'Stage': self.stage, 'Status': 'Complete', 'TargetMonth': self.target_month, 
                'Detail': "{} clusters retained".format(len(retained_clusters)) })

    def _run_factor_optimize(self):

        logger.info("Running factor optimize")

        self._init_settings()

        # Check to see if Optimize is required
        if 'optimize' not in self.settings['FactorSelectionCombinationGenerationMethods']:
            self.db.update_project_run_state(self.project_id, { 'Stage': 'FactorOptimize', 'Status': 'Complete', 'TargetMonth': self.target_month, 'Detail': 'Optimize skipped' })
            return

        self._load_project_data_info()

        # Load returns and factor expected returns
        returns_df, factor_dfs = self._load_expected_returns()

        # Get factor clusters
        cluster_df = self.db.get_project_factor_clusters(self.project_id, self.target_month)

        # Filter clusters to factors with expected returns
        cluster_df = cluster_df.loc[factor_dfs.keys()]

        # Apply factor filter
        filter_state = self.db.get_project_factor_filter_state(self.project_id, self.target_month)
        if filter_state is not None:
            retained_clusters = [ trial['Cluster'] for trial in filter_state['Stages'][filter_state['BestStage']]['Trials'] ]
            cluster_df = cluster_df.loc[cluster_df['Cluster'].isin(retained_clusters)]

        # Check settings
        min_cluster_count = int(self.settings['FactorSelectionOptimizeFactorCountLo'])
        max_cluster_count = int(self.settings['FactorSelectionOptimizeFactorCountHi'])

        cluster_count = len(cluster_df['Cluster'].unique())
        if min_cluster_count > cluster_count:
            self.db.update_project_run_state(self.project_id, { 'Stage': 'FactorOptimize', 'Status': 'Failed', 'TargetMonth': self.target_month, 
                'Detail': "Total number of clusters ({}) < min number of clusters for Optimize ({})".format(cluster_count, min_cluster_count) })
            return

        # Run Optimize

        optimizer = optimize.FactorOptimizer(
            returns_df,
            factor_dfs,
            cluster_df,
            int(self.settings['FactorSelectionSTPeriodDuration']), float(self.settings['FactorSelectionSTPeriodWeight']),
            min_cluster_count, max_cluster_count, 
            (int(self.settings['FactorSelectionLongShortPairsStart']), int(self.settings['FactorSelectionLongShortPairsEnd'])),
            self.settings['FactorSelectionObjective'],
            optimize_directions=(self.settings['DriverMethod'] == 'variable_direction'),
        )

        # Initialise with last combination
        last_combination = self.db.get_project_last_generated_factor_combination(self.project_id, self.target_month, 'optimize')
        if last_combination is not None:
            optimizer.set_starting_point(last_combination)

        # Get number of combinations generated
        initial_combination_count = self.db.get_project_factor_combination_count(self.project_id, self.target_month, 'optimize')

        start_time = time.perf_counter()
        for i in range(0, 100):

            # Run optimization
            optimizer.run(min(self.settings['FactorSelectionOptimizeMinCombinations'], 100))

            combination_count = initial_combination_count + len(optimizer.combinations)
            if last_combination is not None:
                combination_count -= 1

            if combination_count >= self.settings['FactorSelectionOptimizeMinCombinations']:
                break

            elapsed_time = time.perf_counter() - start_time
            if elapsed_time > 30.0:
                break

        # Save combinations to db
        combinations = optimizer.combinations if last_combination is None else optimizer.combinations[1:]
        self.db.add_project_factor_combinations(self.project_id, self.target_month, 'optimize', combinations)

        # Update run state
        combination_count = self.db.get_project_factor_combination_count(self.project_id, self.target_month, 'optimize')
        detail = "{} combinations generated".format(combination_count)
        if combination_count >= self.settings['FactorSelectionOptimizeMinCombinations']:
            self.db.update_project_run_state(self.project_id, { 'Stage': 'FactorOptimize', 'Status': 'Complete', 'TargetMonth': self.target_month, 'Detail': detail })
        else:
            self.db.update_project_run_state(self.project_id, { 'Stage': 'FactorOptimize', 'Status': 'InProgress', 'TargetMonth': self.target_month, 'Detail': detail })


    def _run_factor_generate(self):

        logger.info("Running factor generate")

        self._init_settings()

        # Check to see if Optimize is required
        if 'generate' not in self.settings['FactorSelectionCombinationGenerationMethods']:
            self.db.update_project_run_state(self.project_id, { 'Stage': 'FactorGenerate', 'Status': 'Complete', 'TargetMonth': self.target_month, 'Detail': 'Generate skipped' })
            return

        self._load_project_data_info()

        # Load returns and factor expected returns
        returns_df, factor_dfs = self._load_expected_returns()

        # Get factor clusters
        cluster_df = self.db.get_project_factor_clusters(self.project_id, self.target_month)

        # Filter clusters to factors with expected returns
        cluster_df = cluster_df.loc[factor_dfs.keys()]

        # Apply factor filter
        filter_state = self.db.get_project_factor_filter_state(self.project_id, self.target_month)
        if filter_state is not None:
            retained_clusters = [ trial['Cluster'] for trial in filter_state['Stages'][filter_state['BestStage']]['Trials'] ]
            cluster_df = cluster_df.loc[cluster_df['Cluster'].isin(retained_clusters)]

        # Check settings
        min_cluster_count = int(self.settings['FactorSelectionGenerateFactorCountLo'])
        max_cluster_count = int(self.settings['FactorSelectionGenerateFactorCountHi'])

        cluster_count = len(cluster_df['Cluster'].unique())
        if min_cluster_count > cluster_count:
            self.db.update_project_run_state(self.project_id, { 'Stage': 'FactorGenerate', 'Status': 'Failed', 'TargetMonth': self.target_month, 
                'Detail': "Total number of clusters ({}) < min number of clusters for Generate ({})".format(cluster_count, min_cluster_count) })
            return

        max_cluster_count = min(max_cluster_count, cluster_count)

        # Run Generate

        state = self.db.get_project_factor_generate_state(self.project_id, self.target_month)

        generator = generate.Generator(
            returns_df,
            factor_dfs,
            cluster_df,
            int(self.settings['FactorSelectionSTPeriodDuration']), 
            float(self.settings['FactorSelectionSTPeriodWeight']),
            self.settings['FactorSelectionObjective'],
            (int(self.settings['FactorSelectionLongShortPairsStart']), int(self.settings['FactorSelectionLongShortPairsEnd'])),
            (min_cluster_count, max_cluster_count),
            int(self.settings['FactorSelectionGenerateMaxCombinationsPerStage']),
            int(self.settings['FactorSelectionGenerateTopFactorCount']),
            self.settings['FactorSelectionGenerateTopFactorObjective'],
            optimize_directions=(self.settings['DriverMethod'] == 'variable_direction'),
            state=state,
        )

        start_time = time.perf_counter()
        for i in range(0, 100):

            # Generate next
            combinations = generator.generate_next()

            if len(combinations) > 0:

                # Store combinations
                self.db.add_project_factor_combinations(self.project_id, self.target_month, 'generate', combinations)

            # Update state
            self.db.update_project_factor_generate_state(self.project_id, self.target_month, generator.state)

            elapsed_time = time.perf_counter() - start_time
            if elapsed_time > 30.0:
                break

        # Update run state
        combination_count = self.db.get_project_factor_combination_count(self.project_id, self.target_month, 'generate')
        detail = "{} combinations generated".format(combination_count)
        status = 'Complete' if generator.state['Complete'] else 'InProgress'
        self.db.update_project_run_state(self.project_id, { 'Stage': 'FactorGenerate', 'Status': status, 'TargetMonth': self.target_month, 'Detail': detail })

    def _run_factor_strategies(self):

        self._init_settings()
        self._load_project_data_info()

        contexts = self.settings['FactorSelectionCombinationGenerationMethods']

        pairs_range = (
            self.settings['LongShortPairsTarget']-self.settings['LongShortPairsDelta'], 
            self.settings['LongShortPairsTarget']+self.settings['LongShortPairsDelta']
            )

        # Get strategies
        strategies = self.db.get_factor_strategies(self.settings['FactorSelectionStrategies'])

        # Get factor expected returns
        factor_dfs = self.db.get_project_factor_expected_returns(self.project_id, self.target_month)
        for key, df in factor_dfs.items():
            factor_dfs[key] = df[self.target_month].to_frame(self.target_month)

        # Get enabled stocks and apply to factors
        stock_selection_state = self.db.get_project_stock_selection_state(self.project_id, self.target_month)
        stocks_enabled = stock_selection.StockSelector.get_enabled_stocks_from_state(stock_selection_state, apply_market_cap_filter=(self.stage in self.settings['MarketCapFilterStages']))
        for key, df in factor_dfs.items():
            factor_dfs[key] = df.loc[stocks_enabled]

        # Get returns (unless target month is forward month)
        returns_data = None
        if self.target_month != self.forward_month:
            returns_data = self.db.get_project_data(self.project_id, self.project_data_info, factor_indexes=[])['Returns']
            returns_data = returns_data.loc[self.target_month].to_frame()
            factor_index = list(factor_dfs.values())[0].index
            returns_data = returns_data.loc[factor_index]

        # Get clusters
        cluster_df = self.db.get_project_factor_clusters(self.project_id, self.target_month)

        # Filter clusters to factors with expected returns
        cluster_df = cluster_df.loc[factor_dfs.keys()]

        for context in contexts:

            # Get factor combinations
            context_combinations = self.db.get_project_factor_combinations(self.project_id, self.target_month, context=context)
            if len(context_combinations) == 0:
                continue

            # Make combinations df
            combinations_df = FactorStrategy.make_combinations_df(context_combinations)

            # Run strategies
            for strategy_def in strategies:
                strategy = FactorStrategy(**strategy_def['Definition'])

                result = strategy.calculate(combinations_df, cluster_df, pairs_range, factor_dfs, returns_data=returns_data)

                self.db.add_project_factor_strategy_result(
                    self.project_id, self.target_month, context,
                    strategy_def['ID'], strategy_def['Definition'], strategy_def['Description'],
                    result
                    )

        self.db.update_project_run_state(self.project_id, { 'Stage': 'FactorStrategies', 'Status': 'Complete', 'TargetMonth': self.target_month, 'Detail': "" })


    def _run_ticker_selection(self):

        logger.info("Running ticker selection")

        self._init_settings()

        if self.settings['DriverMethod'] == 'variable_direction':
            self.db.update_project_run_state(self.project_id, { 'Stage': self.stage, 'Status': 'Complete', 'TargetMonth': self.target_month, 
                'Detail': "No Ticker Selection for Variable Direction Driver method" })
            return

        self._load_project_data_info()

        self.db.update_project_run_state(self.project_id, { 'Stage': self.stage, 'Status': 'InProgress', 'TargetMonth': self.target_month, 'Detail': "" })

        # Load returns and factor expected returns
        returns_df, factor_dfs = self._load_expected_returns()

        # Load stock selection state
        state = self.db.get_project_stock_selection_state(self.project_id, self.target_month)

        # Run stock selection
        stock_selector = stock_selection.StockSelector(
            returns_df,
            factor_dfs,
            st_duration=int(self.settings['FactorSelectionSTPeriodDuration']),
            st_weight=float(self.settings['FactorSelectionSTPeriodWeight']),
            objective=self.settings['FactorSelectionObjective'],
            score_pairs_range=(int(self.settings['FactorSelectionLongShortPairsStart']), int(self.settings['FactorSelectionLongShortPairsEnd'])),
            min_stock_count=self.settings['TickerSelectionMinStocks'],
            score_impact_tolerance=self.settings['TickerSelectionScoreImpactTolerance'],
            state=state,
        )

        start_time = time.perf_counter()
        for i in range(0, 1000):

            stock_selector.run()

            if stock_selector.state['Complete']:
                self.db.update_project_stock_selection_state(self.project_id, self.target_month, stock_selector.state)
            
                stocks_removed = [stage['StockRemoved'] for stage in stock_selector.state['Stages'] if stage['StockRemoved'] is not None]
                stocks_removed = [self.project_data_info['Stocks'][stock]['Name'] for stock in stocks_removed]
                detail = "{} stocks removed by performance impact".format(len(stocks_removed))
                if len(stocks_removed) > 0:
                    detail += ": {}".format(", ".join(stocks_removed))
                self.db.update_project_run_state(self.project_id, { 'Stage': 'TickerSelection', 'Status': 'Complete', 'TargetMonth': self.target_month, 'Detail': detail })

                return

            last_stage = stock_selector.state['Stages'][-1]
            detail = "Step {}: Baseline score = {:.3f}".format(len(stock_selector.state['Stages']), last_stage['BaselineScore'])
            if last_stage['Trials'][0]['ScoreDelta'] is None:
                if len(stock_selector.state['Stages']) > 1:
                    last_trial = stock_selector.state['Stages'][-2]['Trials'][-1]
                    stock_removed = stock_selector.state['Stages'][-2]['StockRemoved']
                    detail = "Step {}: Score improvement = {:.3f} for removing stock {}; Removed stock {}".format(
                        len(stock_selector.state['Stages'])-1,
                        last_trial['ScoreDelta'], 
                        self.project_data_info['Stocks'][last_trial['Stock']]['Name'],
                        self.project_data_info['Stocks'][stock_removed]['Name'])
            else:
                for index, trial in enumerate(last_stage['Trials'][1:]):
                    if trial['ScoreDelta'] is None:
                        last_trial = last_stage['Trials'][index]
                        if last_trial['ScoreDelta'] is not None:
                            detail = "Step {}: Score improvement = {:.3f} for removing stock {}".format(
                                len(stock_selector.state['Stages']),
                                last_trial['ScoreDelta'], 
                                self.project_data_info['Stocks'][last_trial['Stock']]['Name'])
                        break

            self.db.update_project_run_state(self.project_id, { 'Stage': 'TickerSelection', 'Status': 'InProgress', 'TargetMonth': self.target_month, 'Detail': detail })

            elapsed_time = time.perf_counter() - start_time
            if elapsed_time > 30.0:
                break
        
        self.db.update_project_stock_selection_state(self.project_id, self.target_month, stock_selector.state)
