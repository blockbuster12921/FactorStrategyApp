import pandas as pd
import numpy as np
from typing import List
import logging
from . import stock_data, factor_calc, score_calc, factor_cluster
from .utils import MeasureTime, MeasureBlockTime

logger = logging.getLogger(__name__)

class FactorOptimizer():
    
    def __init__(self, 
                 returns_df,
                 factor_dfs,
                 cluster_df,
                 st_duration, st_weight,
                 min_clusters, max_clusters, 
                 score_pairs_range,
                 objective,
                 optimize_directions=False,
                 random_seed=None):

        self.returns_df = returns_df
        self.factor_dfs = factor_dfs

        self.clusters = factor_cluster.get_clustered_factors_from_df(cluster_df)
        self.cluster_keys = list(self.clusters.keys())

        self.cluster_expected_returns = {}
        for i, cluster_factors in self.clusters.items():
            if len(cluster_factors) == 1:
                self.cluster_expected_returns[i] = self.factor_dfs[cluster_factors[0]]
            else:
                self.cluster_expected_returns[i] = factor_calc.combine_factors(self.factor_dfs, cluster_factors)

        assert(min_clusters <= len(self.clusters))
        assert(min_clusters <= max_clusters)
        self.min_clusters = min_clusters
        self.max_clusters = min(max_clusters, len(self.clusters))

        assert(st_duration >= 0)
        assert(st_weight >= 0.0)
        self.st_weight = st_weight

        self.st_months = None
        self.st_period_start = None
        if (self.st_weight > 0.0) and (st_duration > 0):
            self.st_months = self.returns_df.columns.tolist()[-st_duration:]
            self.st_period_start = self.st_months[0]

        assert(score_pairs_range[0] <= score_pairs_range[1])
        assert(score_pairs_range[0] >= 1)
        self.score_pairs_range = score_pairs_range

        self.combinations = []

        self.objective = objective

        self.optimize_directions = optimize_directions

        # Random number generator
        self.rng = np.random.RandomState(random_seed)

        # "Temperature" for accept/reject step
        if self.objective == 'score':
            self.temperature = 0.06
        elif self.objective in ['corr', 'rank_corr']:
            self.temperature = 1.0 * np.power(len(self.returns_df.columns)/12.0, -0.7)
        else:
            raise ValueError("Invalid objective method '{}'".format(self.objective))

    def set_starting_point(self, combination):
        assert(len(combination['Clusters']) >= self.min_clusters)
        assert(len(combination['Clusters']) <= self.max_clusters)
        assert(combination.get('Score') is not None)
        self.combinations.append(combination)

    def run(self, trials):

        assert(trials > 0)
        
        if len(self.combinations) == 0:
            # Generate random starting point
            cluster_count = int((self.min_clusters+self.max_clusters)/2)
            shuffled_clusters = list(self.cluster_keys)
            self.rng.shuffle(shuffled_clusters)
            clusters = sorted(shuffled_clusters[:cluster_count])
            if self.optimize_directions:
                # Randomize cluster directions
                for i, f in enumerate(clusters):
                    if self.rng.randint(2) == 0:
                        clusters[i] = -f - 1
            self._calc_combination(clusters)
            trials -= 1

        accepted_combination = self.combinations[-1]
        
        rejected_count = 0

        for step in range(0, trials):
            
            clusters = self._update_clusters(accepted_combination['Clusters'].copy())
            self._calc_combination(clusters)

            score_change = self.combinations[-1]['Score'] - accepted_combination['Score']
            if score_change > 0:
                accepted_combination = self.combinations[-1]
            else:
                prob = np.exp(score_change/self.temperature)
                if self.rng.uniform() < prob:
                    accepted_combination = self.combinations[-1]
                else:
                    rejected_count += 1

            if (step+1) % 100 == 0:
                rejection_rate = rejected_count/(step+1)

        acceptance_rate = 100.0 * (1.0 - rejected_count/(step+1))
        logger.debug("Acceptance rate = {:.2f}%".format(step+1, acceptance_rate))

    def _update_clusters(self, clusters):
        
        assert(len(clusters) >= self.min_clusters)
        assert(len(clusters) <= self.max_clusters)

        def add_cluster(clusters):
            while True:
                new_cluster = self.cluster_keys[self.rng.randint(len(self.cluster_keys))]
                if new_cluster in clusters:
                    continue

                if self.optimize_directions:
                    flipped_cluster = -new_cluster - 1
                    if flipped_cluster in clusters:
                        continue
                    if self.rng.randint(2) == 0:
                        new_cluster = flipped_cluster

                break

            clusters.append(new_cluster)
            return sorted(clusters)

        def remove_cluster(clusters):
            del clusters[self.rng.randint(len(clusters))]
            return clusters
            
        def swap_cluster(clusters):
            index_to_swap = self.rng.randint(len(clusters))

            del clusters[index_to_swap]
            return add_cluster(clusters)

        if self.min_clusters == self.max_clusters:
            return swap_cluster(clusters)

        if len(clusters) == self.min_clusters:
            if self.rng.randint(2) == 0:
                return add_cluster(clusters)
            else:
                return swap_cluster(clusters)

        if len(clusters) == self.max_clusters:
            if len(clusters) == len(self.cluster_keys):
                return remove_cluster(clusters)

            if self.rng.randint(2) == 0:
                return remove_cluster(clusters)
            else:
                return swap_cluster(clusters)

        option = self.rng.randint(3)
        if option == 0:
            return remove_cluster(clusters)
        elif option == 1:
            return add_cluster(clusters)
        else:
            return swap_cluster(clusters)
    
    def _calc_combination(self, clusters):

        factor_av = factor_calc.combine_factors(self.cluster_expected_returns, clusters)

        if self.objective == 'score':

            deltas = factor_calc.calc_long_short_pair_return_deltas(
                factor_av, 
                self.returns_df, 
                self.score_pairs_range[0], self.score_pairs_range[1]
                )

            metrics = factor_calc.calc_long_short_pairs_metrics(deltas)
            score = score_calc.calc_score(metrics)

            if self.st_period_start is not None:
                st_metrics = factor_calc.calc_long_short_pairs_metrics(deltas, start_date=self.st_period_start)
                st_score = score_calc.calc_score(st_metrics)
                score = (score + self.st_weight*st_score)/(1.0 + self.st_weight)

        elif (self.objective == 'corr') or (self.objective == 'rank_corr'):

            corr_method = 'spearman' if self.objective == 'rank_corr' else 'pearson'

            corrs = factor_av.corrwith(self.returns_df, axis=0, method=corr_method)
            score = corrs.mean()

            if self.st_months is not None:
                st_score = corrs.loc[self.st_months].mean()
                score = (score + self.st_weight*st_score)/(1.0 + self.st_weight)

            score *= 100.0

        else:
            raise ValueError("Invalid objective method '{}'".format(self.objective))

        self.combinations.append({
            'Clusters': clusters,
            'Score': score,
            'TimePeriod': (self.returns_df.columns[0], self.returns_df.columns[-1])
        })

        logger.debug("Score {:5f} for {}".format(self.combinations[-1]['Score'], clusters))

        return self.combinations[-1]


