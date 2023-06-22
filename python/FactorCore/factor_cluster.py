import pandas as pd
import numpy as np
import sklearn.metrics.pairwise
import sklearn.cluster
import openpyxl, openpyxl.utils.dataframe
import logging
from .utils import MeasureTime, MeasureBlockTime

logger = logging.getLogger(__name__)

def calc_cluster_sizes_from_df(cluster_df):
    return cluster_df.groupby('Cluster')['Cluster'].transform('count')

def get_clustered_factors_from_df(cluster_df):
    return cluster_df.reset_index().groupby('Cluster')['Factor'].apply(list).to_dict()

def get_cluster_matrix_from_df(cluster_df):
    cluster_matrix = pd.DataFrame(index=cluster_df.index, columns=cluster_df.index, data=np.nan)

    clustered_factors = get_clustered_factors_from_df(cluster_df)

    for c in clustered_factors.values():
        cluster_matrix.loc[cluster_matrix.index.isin(c), c] = 1

    return cluster_matrix

class FactorClusterGenerator():

    def __init__(
        self, 
        returns_df, 
        factor_dfs,
        stocks_enabled,
        start_date,
        end_date,
        distance_threshold_multiplier,
        min_data_fraction=0.1,
        ):
        self.returns_df = returns_df.loc[start_date:end_date][stocks_enabled]
        self.factor_dfs = { f: df.loc[start_date:end_date][stocks_enabled] for f, df in factor_dfs.items() }

        self.distance_threshold_multiplier = distance_threshold_multiplier

        self.min_data_fraction = min_data_fraction

    def run(self):

        if self.distance_threshold_multiplier <= 0.0:
            return self._make_individual_clusters()

        # Calculate correlations and check number of non-null values
        corrs = self._calc_factor_correlations()

        all_factors = corrs.columns

        corr_count = corrs.count(axis=0)
        clustering_factors = corr_count.loc[corr_count > self.min_data_fraction*corr_count.max()].index.tolist()

        if len(clustering_factors) == 0:
            return self._make_individual_clusters()

        corrs = corrs[clustering_factors]

        # Calculate distances
        pdist = sklearn.metrics.pairwise.nan_euclidean_distances(corrs.T)

        # Run clustering
        clustering = sklearn.cluster.AgglomerativeClustering(
            linkage='average', compute_full_tree=True, 
            distance_threshold=self.distance_threshold_multiplier*np.nanmean(pdist), 
            n_clusters=None, affinity='precomputed')
        clustering = clustering.fit(pdist)

        cluster_df = pd.Series(index=corrs.columns, data=clustering.labels_).to_frame('Cluster')
        cluster_df.index.name = 'Factor'
        assert(cluster_df['Cluster'].min() == 0)
        assert(cluster_df['Cluster'].max() == len(cluster_df['Cluster'].unique())-1)

        # Insert remaining factors    
        for f in set(all_factors) - set(clustering_factors):
            cluster_df.at[f, 'Cluster'] = cluster_df['Cluster'].max() + 1

        cluster_df['Cluster'] = cluster_df['Cluster'].astype(int)
        cluster_df = cluster_df.sort_index()

        return cluster_df

    def _make_individual_clusters(self):
        factors = list(self.factor_dfs.keys())
        cluster_df = pd.DataFrame(data={ 'Factor': factors, 'Cluster': list(range(len(factors))) }).set_index('Factor')
        return cluster_df

    def _calc_factor_correlations(self):

        corrs = {}

        for f in self.factor_dfs.keys():
            factor_df = self.factor_dfs[f] 
            corrs[f] = self.returns_df.corrwith(factor_df, axis=1, method='spearman')

        corrs = pd.DataFrame(corrs)
        return corrs

class FactorClusterReportGenerator():

    def __init__(self, db, project_id):
        self.db = db
        self.project_id = project_id
    
    def _write_dataframe(self, ws, df):
        
        for r in openpyxl.utils.dataframe.dataframe_to_rows(df, index=False, header=True):
            ws.append(r)

        ws.auto_filter.ref = ws.dimensions

    def generate(self):

        wb = openpyxl.Workbook(write_only=False)
        wb.remove(wb.active)

        data_info = self.db.get_project_data_info(self.project_id)
        factor_names = data_info['Factors']

        target_month_clusters_list = self.db.get_project_factor_clusters(self.project_id)

        for target_month_clusters in target_month_clusters_list:
            
            month = "{:%b-%Y}".format(target_month_clusters['TargetMonth'])
            cluster_df = target_month_clusters['Clusters']

            cluster_df.index = pd.Index(data=[factor_names[f] for f in cluster_df.index], name=cluster_df.index.name)

            cluster_matrix = get_cluster_matrix_from_df(cluster_df)

            ws = wb.create_sheet(month)
            self._write_dataframe(ws, cluster_matrix.reset_index())

        return wb
