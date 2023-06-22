import pandas as pd
import numpy as np
import logging
from . import factor_calc, stock_selection, utils, settings, factor_cluster

logger = logging.getLogger(__name__)

def calculate_project_factor_strategies(db, project_id, strategy_ids):

    project = db.get_project(project_id)
    data_info = db.get_project_data_info(project_id)
    project_settings = db.get_project_settings(project_id)
    project_settings = settings.overlay_default_project_settings(project_settings)

    # Get forward and target months
    forward_month = pd.Timestamp(data_info['Dates'][-1])

    if project['Status'] == 'Live':
        target_months = [forward_month]
    else:
        assert(project['Status'] == 'Test')
        start_date = pd.Timestamp(project['OOSStartDate'])
        target_months = [pd.Timestamp(m) for m in data_info['Dates'] if (pd.Timestamp(m) >= start_date) and (pd.Timestamp(m) < forward_month)]

    # Get settings
    contexts = project_settings['FactorSelectionCombinationGenerationMethods']

    pairs_range = (
        project_settings['LongShortPairsTarget']-project_settings['LongShortPairsDelta'], 
        project_settings['LongShortPairsTarget']+project_settings['LongShortPairsDelta']
        )

    # Get strategies
    strategies = db.get_factor_strategies(strategy_ids)

    for target_month in target_months:

        # Get factor expected returns
        factor_dfs = db.get_project_factor_expected_returns(project_id, target_month)
        for key, df in factor_dfs.items():
            factor_dfs[key] = df[target_month].to_frame(target_month)

        # Get enabled stocks and apply to factors
        stock_selection_state = db.get_project_stock_selection_state(project_id, target_month)
        stocks_enabled = stock_selection.StockSelector.get_enabled_stocks_from_state(stock_selection_state, apply_market_cap_filter=('FactorStrategies' in project_settings['MarketCapFilterStages']))
        for key, df in factor_dfs.items():
            factor_dfs[key] = df.loc[stocks_enabled]

        # Get returns (unless target month is forward month)
        returns_data = None
        if target_month != forward_month:
            returns_data = db.get_project_data(project_id, data_info, factor_indexes=[])['Returns']
            returns_data = returns_data.loc[target_month].to_frame()
            factor_index = list(factor_dfs.values())[0].index
            returns_data = returns_data.loc[factor_index]

        for context in contexts:

            # Get factor combinations
            context_combinations = db.get_project_factor_combinations(project_id, target_month, context=context)
            if len(context_combinations) == 0:
                continue

            # Make combinations df
            combinations_df = FactorStrategy.make_combinations_df(context_combinations)

            # Run strategies
            for strategy_def in strategies:
                strategy = FactorStrategy(**strategy_def['Definition'])

                result = strategy.calculate(combinations_df, pairs_range, factor_dfs, returns_data=returns_data)

                db.add_project_factor_strategy_result(
                    project_id, target_month, context,
                    strategy_def['ID'], strategy_def['Definition'], strategy_def['Description'],
                    result
                    )


class FactorStrategy():

    def __init__(    
        self,
        equal_weights=False,
        min_score=None,
        top_n_combinations=None,
        top_percent_combinations=None,
        exclude_top_percent_combinations=None,
        top_n_factors=None,
        min_occurrences=None,
        exclude_top_n_factors=0,
    ):
        self.equal_weights = equal_weights
        self.min_score = min_score
        self.top_n_combinations = top_n_combinations
        self.top_percent_combinations = top_percent_combinations
        self.exclude_top_percent_combinations = exclude_top_percent_combinations
        self.top_n_factors = top_n_factors
        self.min_occurrences = min_occurrences
        self.exclude_top_n_factors = exclude_top_n_factors

        assert((top_percent_combinations is None) or ((top_percent_combinations > 0) and (top_percent_combinations <= 100)))
        assert((exclude_top_percent_combinations is None) or ((exclude_top_percent_combinations > 0) and (exclude_top_percent_combinations <= 100)))


    def describe(self):

        combinations = ''
        if self.min_score is not None:
            combinations = 'Score >= {:.12g}'.format(self.min_score)
        if self.top_n_combinations is not None:
            if len(combinations) > 0:
                combinations += ' and '
            combinations += 'Top {}'.format(self.top_n_combinations)
        if self.top_percent_combinations is not None:
            if len(combinations) > 0:
                combinations += ' and '
            combinations += 'Top {:.12g}%'.format(self.top_percent_combinations)
        if self.exclude_top_percent_combinations is not None:
            if len(combinations) > 0:
                combinations += ' '
            combinations += 'ex Top {:.12g}%'.format(self.exclude_top_percent_combinations)
        if len(combinations) == 0:
            combinations = 'All'

        factors = ''
        if self.top_n_factors is not None:
            factors = 'Top {}'.format(self.top_n_factors)
        else:
            factors = 'All'
        if (self.exclude_top_n_factors is not None) and (self.exclude_top_n_factors > 0):
            factors += ' ex Top {}'.format(self.exclude_top_n_factors)
        if (self.min_occurrences is not None) and (self.min_occurrences > 0):
            factors += ' appearing >= {} times'.format(self.min_occurrences)

        weighting = 'Equal' if self.equal_weights else 'Frequency'

        return { 'combinations': combinations, 'factors': factors, 'weighting': weighting }

    def calculate(self, combinations_df, cluster_df, pairs_range, factors_data, returns_data=None):
    
        cluster_weights = self._calc_weights_from_combinations(combinations_df)
        if cluster_weights is None:
            return {
                'Factors': None,
                'FactorWeights': None,
                'RankedStocks': None,
                'ReturnDeltas': None,
            }

        cluster_factors_dict = factor_cluster.get_clustered_factors_from_df(cluster_df)

        factors = []
        factor_weights = []
        for i, cluster in enumerate(cluster_weights['Clusters']):

            cluster_factors = cluster_factors_dict[cluster]

            factor_weight = cluster_weights['Weights'][i] / len(cluster_factors)

            factors += cluster_factors
            factor_weights += [factor_weight] * len(cluster_factors)

        factor_av = factor_calc.combine_factors_weighted(factors_data, factors, factor_weights)

        return_deltas = None
        if returns_data is not None:
            return_deltas = factor_calc.calc_long_short_pair_return_deltas(factor_av, returns_data, pairs_range[0], pairs_range[1])

        ranked_stocks = factor_calc.calc_factor_ranking(factor_av[factor_av.columns[0]])

        return {
            'Factors': factors,
            'FactorWeights': factor_weights,
            'RankedStocks': ranked_stocks,
            'ReturnDeltas': None if return_deltas is None else { 'Pairs': return_deltas.columns.tolist(), 'Values': return_deltas.iloc[0].values.tolist() },
        }

    @staticmethod
    def make_combinations_df(combinations):

        comb_df = []
        scores = []
        for comb in combinations:
            row = { cluster: 1 for cluster in comb['Clusters'] }
            comb_df.append(row)
            scores.append(comb['Score'])

        comb_df = pd.DataFrame(comb_df).fillna(0.0)
        comb_df = comb_df.reindex(sorted(comb_df.columns), axis=1)

        comb_df['Score'] = scores

        return comb_df

    def _calc_weights_from_combinations(self, comb_df):

        top_percent_combinations_percentile = None
        if self.top_percent_combinations is not None:
            top_percent_combinations_percentile = comb_df['Score'].quantile(1.0-self.top_percent_combinations/100.0)

        exclude_top_percent_combinations_percentile = None
        if self.exclude_top_percent_combinations is not None:
            exclude_top_percent_combinations_percentile = comb_df['Score'].quantile(1.0-self.exclude_top_percent_combinations/100.0)

        if top_percent_combinations_percentile is not None:
            comb_df = comb_df.loc[comb_df['Score'] >= top_percent_combinations_percentile]

        if exclude_top_percent_combinations_percentile is not None:
            comb_df = comb_df.loc[comb_df['Score'] <= exclude_top_percent_combinations_percentile]

        if self.min_score is not None:
            comb_df = comb_df.loc[comb_df['Score'] >= self.min_score]

        if self.top_n_combinations is not None:
            comb_df = comb_df.nlargest(self.top_n_combinations, 'Score', keep='all')

        if len(comb_df.index) == 0:
            logger.debug("No filtered combinations found")
            return None

        frequencies = comb_df.sum(axis=0).iloc[:-1].sort_values(ascending=False)
        frequencies = frequencies.loc[frequencies > 0]
        frequencies = frequencies.astype('float')

        # Keep factor direction with higher frequency
        sorted_factors = frequencies.index.tolist()
        if min(sorted_factors) < 0:
            retained_factors = []
            for f in sorted_factors:
                flipped = -f-1
                if flipped not in retained_factors:
                    retained_factors.append(f)
            frequencies = frequencies.loc[retained_factors]

        if self.top_n_factors is not None:
            frequencies = frequencies.nlargest(self.top_n_factors, keep='all')
    
        if self.min_occurrences is not None:
            frequencies = frequencies.loc[frequencies >= self.min_occurrences]
        
        if self.exclude_top_n_factors > 0:
            frequencies = frequencies.nsmallest(len(frequencies)-self.exclude_top_n_factors, keep='all')
        
        if len(frequencies) == 0:
            logger.debug("No filtered factors")
            return None

        if self.equal_weights:
            weights = np.ones(len(frequencies)) / len(frequencies)
        else:
            weights = frequencies.values / frequencies.sum()

        return { 
            'Clusters': frequencies.index.to_list(),
            'Frequencies': frequencies.values,
            'Weights': weights,
            }


