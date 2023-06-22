import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calc_score(returns):
    score = 100.0 * np.nanmean(returns)
    score += 2.0 * np.nanmean(returns >= 0)
    score += 100.0 * np.nanmean(returns*(returns < 0))
    return score

def calc_sharpe(x):
    n = len(x)
    return np.sqrt(12.0) * np.nanmean(x) / np.nanstd(x, ddof=1)

def load_backtest_data_from_excel(report_info):

    dfs = []
    for info in report_info:
        for source in ['Opt','Gen']:
            sheet = "{} Returns".format(source)
            logger.info("Loading sheet '{}' from '{}'...".format(sheet, info['Path']))
            df = pd.read_excel(info['Path'], sheet_name=sheet, usecols='A:BP')
            df.insert(0, 'Source', source)
            df.insert(0, 'MC', info['MC'])
            df.insert(0, 'MT', info['MT'])
            dfs.append(df)

    df = pd.concat(dfs, ignore_index=True, sort=False)

    cols = []
    for col in df.columns:
        try:
            month = pd.Timestamp(col)
        except:
            cols.append(col)
        else:
            cols.append(month)
    df.columns = cols

    return df

class Backtester():

    def __init__(self, data):
        
        self.data = data
        
        self.pairs = sorted(self.data['Pairs'].unique().tolist())

        self.all_months = [col for col in self.data.columns if isinstance(col, pd.Timestamp)]
        self.test_months = [month for month in self.all_months if month.year >= 2016]
        
        index_cols = ['MT','MC','Source','Strategy','Pairs']
        self.returns_df = self.data[index_cols+self.all_months].set_index(index_cols)

        self.order_dfs = {}
    
    def run_one(self, order_method, order_months, mt_filter=[], mc_filter=[], source_filter=[]):
        
        key = (order_method, order_months)
        order_df = self.order_dfs.get(key)
        if order_df is None:
            logger.info("Calculating order for method {} with timescale {}".format(order_method, order_months))
            order_df = self.calc_order_data(order_method, order_months)
            self.order_dfs[key] = order_df

        if len(mt_filter) > 0:
            idx = pd.IndexSlice
            order_df = order_df.loc[idx[mt_filter,:,:,:],:]
        if len(mc_filter) > 0:
            idx = pd.IndexSlice
            order_df = order_df.loc[idx[:,mc_filter,:,:],:]
        if len(source_filter) > 0:
            idx = pd.IndexSlice
            order_df = order_df.loc[idx[:,:,source_filter,:],:]

        if len(order_df) == 0:
            return None
        
        idxmax = order_df.idxmax()

        results = {}
        for month in self.test_months:

            top_index = idxmax.at[month]

            pair_results = {}
            for pair in self.pairs:
                idx = (top_index[0], top_index[1], top_index[2], top_index[3], pair)            
                pair_results[pair] = self.returns_df.loc[idx][month]
            results[month] = pair_results

        results = pd.DataFrame(results).T

        results['Pairs Average'] = results.mean(axis=1)
        results['Best Pair'] = results.max(axis=1)
        results['Worst Pair'] = results.min(axis=1)

        return results

    def run_all_methods(self, mt_filter=[], mc_filter=[], source_filter=[], plot_metric=None, plot_pairs="Pairs Average"):

        results = []

        for method in ['Mean','Score','Sharpe']:
            for months in range(2,13):
                test_df = self.run_one(method, months, mt_filter=mt_filter, mc_filter=mc_filter, source_filter=source_filter)
                if test_df is not None:
                    results.append({'Months': months, 'Method': method, 'Metric': 'Mean', **test_df.mean()})
                    results.append({'Months': months, 'Method': method, 'Metric': 'Score', **test_df.apply(calc_score)})
                    results.append({'Months': months, 'Method': method, 'Metric': 'Sharpe', **test_df.apply(calc_sharpe)})

        for method in ['ExpWeight','LinearWeight','ExpWeightScore']:
            for months in (list(range(2, 17)) + [18,20,24,30,36]):
                test_df = self.run_one(method, months, mt_filter=mt_filter, mc_filter=mc_filter, source_filter=source_filter)
                if test_df is not None:
                    results.append({'Months': months, 'Method': method, 'Metric': 'Mean', **test_df.mean()})
                    results.append({'Months': months, 'Method': method, 'Metric': 'Score', **test_df.apply(calc_score)})
                    results.append({'Months': months, 'Method': method, 'Metric': 'Sharpe', **test_df.apply(calc_sharpe)})

        if len(results) == 0:
            return None

        result_df = pd.DataFrame(results)

        def init_filter_cols(result_df, filter_col, filter_list):
            all_vals = sorted(self.data[filter_col].unique().tolist())
            cols = { val: (len(filter_list) == 0) for val in all_vals }
            for val in filter_list:
                cols[val] = True
            for key, val in cols.items():
                result_df["{}={}".format(filter_col, key)] = val

        col_count = len(result_df.columns)

        for filter_col, filter_list in [('MT',mt_filter), ('MC',mc_filter), ('Source',source_filter)]:
            init_filter_cols(result_df, filter_col, filter_list)

        cols = result_df.columns.tolist()
        cols = cols[col_count:] + cols[:col_count]
        result_df = result_df[cols]

        if plot_metric is not None:
            plot_df = result_df.loc[result_df.Metric==plot_metric][['Months','Method',plot_pairs]].set_index(['Months','Method']).unstack()
            plot_df.columns = plot_df.columns.droplevel(0)
            title = "MT={}, MC={}, Source={}".format(mt_filter, mc_filter, source_filter)
            plot_df.plot(figsize=(16,6), title=title)

        return result_df

    def run_all_scenarios(self):

        filters = []
        perm_count = 0
        for f in ['MT','MC','Source']:
            filters.append( (f, sorted(self.data[f].unique().tolist())) )
            perm_count += len(filters[-1][1])

        perm_count = 2**perm_count

        perms = []
        for i in range(1, perm_count):
            mt = []
            mc = []
            sources = []
            for f in filters:
                for val in f[1]:
                    if (i % 2) == 1:
                        if f[0] == 'MT':
                            mt.append(val)
                        elif f[0] == 'MC':
                            mc.append(val)
                        else:
                            sources.append(val)
                    i = int(i/2)
            if (len(mt) > 0) & (len(mc) > 0) & (len(sources) > 0):
                perms.append([mt, mc, sources])

        results = []
        for perm in perms:
            logger.info("Calculating MT={}, MC={}, Source={}".format(perm[0], perm[1], perm[2]))
            result_df = self.run_all_methods(mt_filter=perm[0], mc_filter=perm[1], source_filter=perm[2])
            if result_df is not None:
                results.append(result_df)

        result_df = pd.concat(results, sort=False)

        return result_df

    def calc_order_data(self, method, timescale):

        if method == 'Mean':
            order_df = self.returns_df.T.rolling(window=timescale, min_periods=min(12,timescale)).mean().shift(1).T
        elif method == 'Score':
            order_df = self.returns_df.T.rolling(window=timescale, min_periods=min(12,timescale)).apply(calc_score, raw=True, engine='numba').shift(1).T
        elif method == 'Sharpe':
            order_df = self.returns_df.T.rolling(window=timescale, min_periods=min(12,timescale)).apply(calc_sharpe, raw=True, engine='numba').shift(1).T
        elif method == 'ExpWeight':

            alpha = 2.0/(1+timescale)
            alpha_prime = 1.0 - alpha
            def weighted_mean(x):
                w = 1.0
                total = x[-1]
                w_total = 1.0
                for i in range(len(x)-2,-1,-1):
                    w *= alpha_prime
                    total += w * x[i]
                    w_total += w
                return total / w_total

            order_df = self.returns_df.T.rolling(window=12, min_periods=12).apply(weighted_mean, raw=True, engine='numba').shift(1).T
        elif method == 'LinearWeight':

            delta = 1.0 / timescale
            def weighted_mean(x):
                w = 1.0
                total = x[-1]
                w_total = 1.0
                for i in range(len(x)-2,-1,-1):
                    w -= delta
                    if w <= 0.0:
                        break
                    total += w * x[i]
                    w_total += w
                return total / w_total

            order_df = self.returns_df.T.rolling(window=12, min_periods=12).apply(weighted_mean, raw=True, engine='numba').shift(1).T
        elif method == 'ExpWeightScore':

            alpha = 2.0/(1+timescale)
            alpha_prime = 1.0 - alpha
            def exp_weighted_score(x):
                w = 1.0
                total = 100.0*x[-1] + 2.0*(x[-1]>=0) + 100.0*(x[-1]*(x[-1]<0))
                w_total = 1.0
                for i in range(len(x)-2,-1,-1):
                    w *= alpha_prime
                    total += w * (100.0*x[i] + 2.0*(x[i]>=0) + 100.0*(x[i]*(x[i]<0)))
                    w_total += w
                return total / w_total

            order_df = self.returns_df.T.rolling(window=12, min_periods=12).apply(exp_weighted_score, raw=True, engine='numba').shift(1).T
        else:
            raise ValueError("Invalid order method '{}'".format(method))
            
        return order_df

