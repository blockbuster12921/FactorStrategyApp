import pandas as pd
import numpy as np
from typing import List
import logging
from . import factor_calc, score_calc, factor_cluster
from .utils import MeasureTime, MeasureBlockTime

logger = logging.getLogger(__name__)

class Generator():

    def __init__(self, 
                 returns_df,
                 factor_dfs,
                 cluster_df,
                 st_duration, 
                 st_weight,
                 objective,
                 score_pairs_range,
                 cluster_count_range,
                 max_combinations,
                 top_cluster_count,
                 top_cluster_objective,
                 optimize_directions=False,
                 state = None,
                 ):
        self.returns_df = returns_df
        self.factor_dfs = factor_dfs

        self.clusters = factor_cluster.get_clustered_factors_from_df(cluster_df)

        self.cluster_expected_returns = {}
        for i, cluster_factors in self.clusters.items():
            if len(cluster_factors) == 1:
                self.cluster_expected_returns[i] = self.factor_dfs[cluster_factors[0]]
            else:
                self.cluster_expected_returns[i] = factor_calc.combine_factors(self.factor_dfs, cluster_factors)

        assert(st_duration >= 0)
        assert(st_weight >= 0.0)
        self.st_weight = st_weight

        self.st_months = None
        self.st_period_start = None
        if (self.st_weight > 0.0) and (st_duration > 0):
            self.st_months = self.returns_df.columns.tolist()[-st_duration:]
            self.st_period_start = self.st_months[0]

        self.objective = objective

        self.score_pairs_range = score_pairs_range

        assert(cluster_count_range[0] >= 1)
        assert(cluster_count_range[1] <= len(self.clusters))
        self.cluster_count_range = cluster_count_range

        self.max_combinations = max_combinations

        self.top_cluster_count = top_cluster_count
        self.top_cluster_objective = top_cluster_objective

        self.optimize_directions = optimize_directions

        self.state = state
        
    # Returns combinations generated
    # Modifies self.state
    def generate_next(self):

        if self.state is None:
            self.state = {
                'Complete': False,
                'BaseCombinations': [[]],
                'PendingCombinations': []
            }

        assert(self.state.get('Complete') is not None)
        assert(self.state.get('BaseCombinations') is not None)
        assert(self.state.get('PendingCombinations') is not None)

        if self.state['Complete']:
            return []

        # Pop first base combination
        base_combination = self.state['BaseCombinations'].pop(0)
        logger.debug("Using base combination {}".format(base_combination))

        # Generate combinations for base and add to pending
        clusters_list, cluster_scores = self._generate_for_base(base_combination)
        logger.debug("Generated {} clusters: {}".format(len(clusters_list), clusters_list))
        for f in clusters_list:
            clusters = sorted(base_combination + [f])
            score = cluster_scores[f]
            exists = False
            for comb in self.state['PendingCombinations']:
                if comb['Clusters'] == clusters:
                    exists = True
                    break
            if not exists:
                self.state['PendingCombinations'].append({'Clusters': clusters, 'Score': score})

        # Check if there are any more base combinations
        if len(self.state['BaseCombinations']) > 0:
            return []

        # Process pending combinations
        logger.debug("No more base combinations - processing pending combinations")

        pending_combinations = self.state['PendingCombinations']
        if len(pending_combinations) == 0:
            raise Exception("No pending combinations found")

        top_pending_combinations = sorted(pending_combinations, key=lambda x: x['Score'], reverse=True)[:self.max_combinations]
        generated_combinations = []

        for combination in top_pending_combinations:
            self.state['BaseCombinations'].append(combination['Clusters'])
            if len(combination['Clusters']) >= self.cluster_count_range[0]:
                generated_combinations.append(combination)
        self.state['PendingCombinations'] = []

        # Check for completion
        if len(top_pending_combinations[0]['Clusters']) == self.cluster_count_range[1]:
            logger.debug("Pending combinations have max number of clusters {} - generation is complete".format(self.cluster_count_range[1]))        
            self.state['Complete'] = True

        return generated_combinations

       
    def _generate_for_base(self, base_combination):

        base_metrics = None
        if len(base_combination) > 0:
            base_metrics, base_score = self._calc_metrics(base_combination)

        combination_metrics = []

        combination_scores = []

        for cluster, cluster_factors in self.clusters.items():
            flipped_cluster = -cluster - 1
            if (cluster in base_combination) or (flipped_cluster in base_combination):
                # Skip clusters already in the combination
                continue

            metrics, score = self._calc_metrics(base_combination + [cluster])
            metrics['Cluster'] = cluster
            combination_metrics.append(metrics)    
            combination_scores.append({'Score': score, 'Cluster': cluster})

            if self.optimize_directions:
                metrics, score = self._calc_metrics(base_combination + [flipped_cluster])
                metrics['Cluster'] = flipped_cluster
                combination_metrics.append(metrics)    
                combination_scores.append({'Score': score, 'Cluster': flipped_cluster})

        # Calculate change in metrics from base combination and convert to Z-Scores
        
        metrics_df = pd.DataFrame(combination_metrics).set_index('Cluster')

        if base_metrics is not None:
            base = pd.Series(base_metrics)
            metrics_df = metrics_df.subtract(base, axis=1)

        # Convert metrics to Z-scores
        metrics_df = (metrics_df - metrics_df.mean()) / metrics_df.std(ddof=0)
        metrics_df = metrics_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        zscores = metrics_df.mean(axis=1)
        
        # Select top cluster
        zscores = zscores.sort_values(ascending=False)
        
        top_clusters = []
        for cluster in zscores.index:
            flipped_cluster = -cluster-1
            if flipped_cluster in top_clusters:
                continue
            top_clusters.append(cluster)
            if len(top_clusters) == self.top_cluster_count:
                break
        
        combination_scores = pd.DataFrame(combination_scores).set_index('Cluster').loc[top_clusters]

        return top_clusters, combination_scores['Score'].to_dict()
    
    def _calc_metrics(self, clusters):
        
        factor_av = factor_calc.combine_factors(self.cluster_expected_returns, clusters)

        if (self.objective == 'score') or (self.top_cluster_objective == 'zscore'):
            deltas = factor_calc.calc_long_short_pair_return_deltas(factor_av, self.returns_df, 
                self.score_pairs_range[0], self.score_pairs_range[1])

            metrics = factor_calc.calc_long_short_pairs_metrics(deltas)

        if self.objective == 'score':
            score = score_calc.calc_score(metrics)

            if self.st_weight > 0.0:
                st_metrics = factor_calc.calc_long_short_pairs_metrics(deltas, start_date=self.st_period_start)
                st_score = score_calc.calc_score(st_metrics)
                score = (score + self.st_weight*st_score)/(1.0 + self.st_weight)

        elif self.objective in ['corr','rank_corr']:
            corr_method = 'spearman' if self.objective == 'rank_corr' else 'pearson'
            corrs = factor_av.corrwith(self.returns_df, axis=0, method=corr_method)

            score = corrs.mean()

            if self.st_months is not None:
                st_score = corrs.loc[self.st_months].mean()
                score = (score + self.st_weight*st_score)/(1.0 + self.st_weight)
            score *= 100.0

        else:
            raise ValueError("Invalid objective method '{}'".format(self.objective))


        top_cluster_metrics = None

        if self.top_cluster_objective == 'zscore':
            zscore_metrics = {}
            zscore_metrics['Sum'] = 100.0 * metrics['Sum'].mean()
            zscore_metrics['PercentPositive'] = 100.0 * metrics['FractionPositive'].mean()
            zscore_metrics['SumNegative'] = 100.0 * metrics['SumNegative'].mean()
            zscore_metrics['MinusCountNegative'] = -metrics['CountNegative'].mean()
            top_cluster_metrics = zscore_metrics

        elif self.top_cluster_objective in ['corr','rank_corr']:

            if self.top_cluster_objective != self.objective:
                corr_method = 'spearman' if self.top_cluster_objective == 'rank_corr' else 'pearson'
                corrs = factor_av.corrwith(self.returns_df, axis=0, method=corr_method)

            top_cluster_metrics = { 'Corr': 100.0 * corrs.mean() }

        else:
            raise ValueError("Invalid top cluster objective method '{}'".format(self.top_cluster_objective))

        return (top_cluster_metrics, score)
    
