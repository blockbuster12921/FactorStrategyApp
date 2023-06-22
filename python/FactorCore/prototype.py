import pandas as pd
import numpy as np
import datetime
import logging
import sklearn
import sklearn.metrics.pairwise
import sklearn.cluster
import scipy.stats
import statsmodels.api as sm
import openpyxl

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from FactorCore import database, utils, driver, factor_calc, score_calc, stock_selection, settings

factor_groupings = {
    'T1': ['F2','F3','F4','F5','F6','F36','F37','F38','F39','F40','F41','F42','F43','F44','F45','F46','F47','F48','F49',
            'F50','F51','F52','F53','F54','F55','F56','F57','F58','F59','F60','F61','F62','F63','F64','F65','F66','F67',
            'F68','F69','F70','F71','F72','F73','F74','F75','F76','F77','F78'],
    'T2': ['F2','F3','F4','F5','F6','F36','F37','F38','F39','F40','F41','F42','F43','F44','F45','F46','F47','F48','F49',
           'F50','F51','F52','F53','F54','F61','F62','F63','F64','F65','F66','F67','F68','F69','F70','F71','F72','F73','F74','F75','F76','F77','F78'],
    'T3': ['F2','F3','F4','F5','F6','F46','F47','F48','F49','F50','F51','F52','F53','F54','F61','F62','F63','F64','F65','F66','F67','F68','F69','F70',
           'F71','F72','F73','F74','F75','F76','F77','F78'],
    'T4': ['F2','F3','F4','F5','F6','F61','F62','F63','F64','F65','F66','F67','F68','F69','F70','F71','F72','F73','F74','F75','F76','F77','F78'],
    'T4&6': ['F2','F3','F4','F5','F6','F23','F24','F27','F49','F50','F59','F61','F62','F63','F64','F65','F66','F67','F68','F69','F70','F71',
               'F72','F73','F74','F75','F76','F77','F78','F81','F84','F88','F91','F95','F98','F105','F109','F113','F117','F121','F125','F129'],
}

def get_factors_not_in_grouping(project, grouping_label):
    return [f for f in project['DataInfo']['Factors'] if f not in factor_groupings[grouping_label]]

def disable_stocks(project, stocks):
    project['Data']['Returns'][stocks] = np.nan
    project['StocksDisabled'] += stocks
    project['StocksDisabled'] = list(set(project['StocksDisabled']))

def load_projects(db, live_only=True, names=None):

    projects = []

    query = { 'Deleted':None }
    if names is not None:
        query['Name'] = { '$in': names }
    if live_only:
        query['Status'] = 'Live'

    for project in db.mongo['Projects'].find(query):

        print("Loading '{}'".format(project['Name']))

        projects.append(project)

        data_info = db.get_project_data_info(project['ID'])
        project['DataInfo'] = data_info
        
        data = db.get_project_data(project['ID'], data_info)
        project['Data'] = data

        project['StocksDisabled'] = []
        stocks_disabled = stock_selection.get_indexes_for_disabled_stocks(data_info['Stocks'], db.get_project_stocks_disabled(project['ID']))
        disable_stocks(project, stocks_disabled)

        # Set returns date ranges
        returns_df = project['Data']['Returns']

        has_return_cumsum = returns_df.notnull().cumsum()
        for stock in has_return_cumsum.columns:
            s = has_return_cumsum[stock].loc[has_return_cumsum[stock] == 1]
            project['DataInfo']['Stocks'][stock]['FirstReturnDate'] = s.index[0] if len(s) > 0 else np.nan
        project['FirstReturnDates'] = pd.Series({ i: stock['FirstReturnDate'] for i, stock in enumerate(project['DataInfo']['Stocks'])})

        has_return_cumsum = returns_df[::-1].notnull().cumsum()
        for stock in has_return_cumsum.columns:
            s = has_return_cumsum[stock].loc[has_return_cumsum[stock] == 1]
            project['DataInfo']['Stocks'][stock]['LastReturnDate'] = s.index[0] if len(s) > 0 else np.nan
        project['LastReturnDates'] = pd.Series({ i: stock['LastReturnDate'] for i, stock in enumerate(project['DataInfo']['Stocks'])})

        project['Settings'] = db.get_project_settings(project['ID'])
        if project['Settings'] is None:
            project['Settings'] = {}
        project['Settings'] = settings.overlay_default_project_settings(project['Settings'])


    return projects

def apply_market_cap_filter_to_returns(project):

    filter_value = project['Settings']['MarketCapFilterValue']
    filter_date = pd.Timestamp(project['Settings']['MarketCapFilterDate'])
    
    stocks_enabled = list(set(range(0, len(project['DataInfo']['Stocks']))) - set(project['StocksDisabled']))

    market_cap_df = project['Data']['Factors'][project['DataInfo']['MarketCapFactorIndex']]
    market_cap_df = market_cap_df[stocks_enabled]

    reference_market_cap = market_cap_df.loc[filter_date]
    
    retained_count = len(reference_market_cap.loc[reference_market_cap >= filter_value])

    returns_df = project['Data']['Returns'].copy().T
    returns_df.loc[~returns_df.index.isin(stocks_enabled)] = np.nan

    for month in returns_df.columns:
        retained_stocks = market_cap_df.loc[month].nlargest(retained_count, keep='all').index.tolist()
        returns_df.loc[~returns_df.index.isin(retained_stocks), month] = np.nan

    return returns_df

def calc_performance(project, factor_weights, top_n_factors, 
                     start_date=None, end_date=None,
                     scale_by_weight=False,
                     plot=False):

    stocks_enabled_count = len(project['DataInfo']['Stocks'])-len(project['StocksDisabled'])
    pairs_max = max(4, min(20, int(0.2*stocks_enabled_count)))

    
    top_factors = factor_weights.abs().sort_values(ascending=False).index[:top_n_factors]

    expected_ranks = {}    
    for factor_index in top_factors:
        
        weight = factor_weights.loc[factor_index]
        
        factor_df = project['Data']['Factors'][factor_index].copy().loc[start_date:end_date]
        factor_df = factor_df.mul(np.sign(weight), axis=0)

        factor_df = factor_df.rank(pct=True, axis=1)

        if scale_by_weight:
            factor_df = factor_df.mul(np.abs(weight), axis=0)

        expected_ranks[factor_index] = factor_df.T

    factor_av = factor_calc.combine_factors(expected_ranks, top_factors)

    returns_df = project['Data']['Returns'].T[factor_av.columns]

    factor_av += 0 * returns_df
    factor_av = factor_av.dropna(how='all', axis=1)
    
    sorted_returns_df = factor_calc.sort_returns_by_factor(factor_av, returns_df, ascending=False)

    deltas = factor_calc.calc_long_short_pair_return_deltas(factor_av, returns_df, 1, pairs_max)
    deltas = deltas.dropna(how='all')
    metrics = factor_calc.calc_long_short_pairs_metrics(deltas)
    metrics.index.name = 'Pairs'

    corrs = returns_df.corrwith(factor_av, method='spearman')
    corrs = corrs.to_frame('Correlation')
    corrs.index.name = 'Date'
    corrs = corrs.reset_index()

    perf = { 'Project': project,
             'Metrics': metrics, 
             'ReturnDeltas': deltas, 
             'Correlations': corrs, 
             'RankedReturns': sorted_returns_df }

    if plot:
        print("{} ({} stocks enabled)".format(
            project['Name'], 
            len(project['DataInfo']['Stocks'])-len(project['StocksDisabled'])))

        plot_performance(perf)

    return perf

def calc_rolling_performance(project, rolling_factor_weights, 
                             start_date, end_date,
                             top_n_factors=None,
                             min_factor_weight=0,
                             factor_direction=0,
                             oos=True,
                             scale_by_weight=None,
                             exclude_stocks=None,
                             ):

    end_date = pd.Timestamp(end_date)
    
    if oos:
        rolling_factor_weights = rolling_factor_weights.shift(1)

    factor_av = {}
    
    factors_selected = {}

    month = pd.Timestamp(start_date)
    while True:

        factor_weights = rolling_factor_weights.loc[month].dropna()
        
        if factor_direction > 0:
            factor_weights = factor_weights.loc[factor_weights > 0]
        elif factor_direction < 0:
            factor_weights = factor_weights.loc[factor_weights < 0]

        if len(factor_weights) > 0:
            
            if min_factor_weight > 0:
                factor_weights = factor_weights.loc[
                    (factor_weights >= min_factor_weight)|(factor_weights <= -min_factor_weight)]

            if top_n_factors is not None:
                top_factors = factor_weights.abs().sort_values(ascending=False).index[:top_n_factors]
            else:
                top_factors = factor_weights.index

            factors_selected[month] = list(top_factors)
    
            if len(top_factors) == 0:
                raise ValueError("No top factors for {}".format(month))

            expected_ranks = {}    
            for factor_index in top_factors:

                weight = factor_weights.loc[factor_index]

                factor_df = project['Data']['Factors'][factor_index].loc[month].copy().to_frame(month)

                factor_df = factor_df.mul(np.sign(weight), axis=0)

                factor_df = factor_df.rank(pct=True, axis=0)

                if scale_by_weight is not None:
                    factor_df = factor_df.mul((np.abs(weight)-min_factor_weight)**scale_by_weight, axis=0)

                expected_ranks[factor_index] = factor_df.T

            factor_av[month] = factor_calc.combine_factors(expected_ranks, top_factors).T[month]
        
        month = (month + pd.DateOffset(months=1)) + pd.offsets.MonthEnd(0)
        if month > end_date:
            break


    factor_av = pd.DataFrame(factor_av)

    returns_df = apply_market_cap_filter_to_returns(project)

    if exclude_stocks is not None:
        returns_df.loc[returns_df.index.isin(exclude_stocks)] = np.nan

    factor_av += 0 * returns_df
    factor_av = factor_av.dropna(how='all', axis=1)
    
    sorted_returns_df = factor_calc.sort_returns_by_factor(factor_av, returns_df, ascending=False)

    stocks_enabled_average = sorted_returns_df.count().mean()
    pairs_max = max(2, min(20, int(0.35*stocks_enabled_average)))

    deltas = factor_calc.calc_long_short_pair_return_deltas(factor_av, returns_df, 1, pairs_max)
    deltas = deltas.dropna(how='all')
    metrics = factor_calc.calc_long_short_pairs_metrics(deltas)
    metrics.index.name = 'Pairs'

    corrs = returns_df.corrwith(factor_av, method='spearman')
    corrs = corrs.to_frame('Correlation')
    corrs.index.name = 'Date'
    corrs = corrs.reset_index()

    ranked_stocks = {}
    for month in factor_av.columns:
        ranked_stocks[month] = factor_calc.calc_factor_ranking(factor_av[month])

    perf = { 'Project': project,
             'TargetReturns': returns_df,
             'CombinedFactors': factor_av,
             'Metrics': metrics, 
             'ReturnDeltas': deltas,
             'Correlations': corrs, 
             'RankedReturns': sorted_returns_df,
             'RankedStocks': ranked_stocks,
             'FactorsSelected': factors_selected,
            }

    return perf

def calc_single_factor_performance(
        project,
        rolling_factor_weights,
        start_date,
        end_date,
        exclude_stocks=None,
    ):

    sharpes = {}
    mean_returns = {}

    for f in rolling_factor_weights.columns:
        perf = calc_rolling_performance(
                            project, rolling_factor_weights[[f]], 
                            top_n_factors=None,
                            min_factor_weight=0,
                            start_date=start_date, end_date=end_date,
                            factor_direction=0,
                            oos=True,
                            scale_by_weight=None,
                            exclude_stocks=exclude_stocks,
                        )
        sharpes[f] = perf['Metrics']['Sharpe']
        mean_returns[f] = perf['Metrics']['Mean']
    
    return { 'Sharpe': pd.DataFrame(sharpes), 'MeanReturn': pd.DataFrame(mean_returns) }
    


def plot_performance(perf):

    min_pairs = 1 if len(perf['Metrics'].index) <= 4 else 2

    project = perf['Project']
    metrics = perf['Metrics'].loc[min_pairs:].reset_index()
    deltas = perf['ReturnDeltas']
    corrs = perf['Correlations']
    sorted_returns_df = perf['RankedReturns']
    pairs_max = deltas.columns[-1]

    average_ranked_returns = sorted_returns_df.median(axis=1)
    ranked_returns_corr = -np.corrcoef(average_ranked_returns.index, average_ranked_returns.values)[0][1]
    print("Ranked returns correlation: {:.1%}".format(ranked_returns_corr))

    display((100*perf['Correlations']['Correlation']).describe().to_frame('Monthly Correlation').T.style.format("{:.3g}"))

    display(perf['Metrics'].style.format(
    { 'Count': '{:.0f}', 'Sum': '{:.1%}', 'Mean': '{:.2%}', 'FractionPositive': '{:.1%}',
      'SumNegative': '{:.1%}', 'MeanNegative': '{:.2%}', 'Sharpe': '{:.2f}'})
       )

    display(perf['ReturnDeltas'].groupby(perf['ReturnDeltas'].index.year).sum().style.format('{:.1%}'))

    fig, axes = plt.subplots(2, 2, figsize=(16,10))
    sns.lineplot(data=metrics, x='Pairs', y='Sharpe', marker='o', ax=axes[0,0])

    plot = sns.lineplot(data=metrics, x='Pairs', y='Mean', marker='o', ax=axes[0,1])
    plot.set_yticklabels(['{:.1%}'.format(x) for x in plot.get_yticks()])
    
    plot = sns.lineplot(data=metrics, x='Pairs', y='FractionPositive', marker='o', ax=axes[1,0])
    plot.set_yticklabels(['{:.0%}'.format(x) for x in plot.get_yticks()])
    plot.set(ylabel='% Positive')

    plot = sns.lineplot(data=metrics, x='Pairs', y='MeanNegative', marker='o', ax=axes[1,1])
    plot.set_yticklabels(['{:.1%}'.format(x) for x in plot.get_yticks()])

    fig, axes = plt.subplots(4, 1, figsize=(16,20))
    corrs_percent_positive = len(corrs.loc[corrs['Correlation'] > 0.0])/len(corrs.loc[corrs['Correlation'].notnull()])
    corrs['Cumulative Correlation'] = corrs['Correlation'].cumsum()
    plot = sns.lineplot(data=corrs, x='Date', y='Cumulative Correlation', ax=axes[0])
    plot.set(title="Cumulative Correlation ({:.1%} positive)".format(corrs_percent_positive))

    corrsign = corrs.copy()
    corrsign['Correlation'] = np.sign(corrsign['Correlation'])
    plot = sns.lineplot(data=corrsign, x='Date', y='Correlation', ax=axes[1])
    plot.set(ylabel='Correlation Sign')

    cum_returns = deltas[range(min_pairs,pairs_max+1)].cumsum()
    cum_returns = cum_returns.stack().to_frame().reset_index()
    cum_returns.columns = ['Date','Pairs','Cumulative Return']
    plot = sns.lineplot(data=cum_returns, x='Date', y='Cumulative Return', hue='Pairs', palette='rocket', 
                        ax=axes[2])
    plot.set_yticklabels(['{:.0%}'.format(x) for x in plot.get_yticks()])

    returns = deltas[range(min_pairs,pairs_max+1)]
    returns = returns.stack().to_frame().reset_index()
    returns.columns = ['Date','Pairs','Return']
    plot = sns.lineplot(data=returns, x='Date', y='Return', hue='Pairs', palette='rocket', ax=axes[3])
    plot.set_yticklabels(['{:.0%}'.format(x) for x in plot.get_yticks()])


def calc_factor_weightings(project, start_date, end_date, 
                           method,
                           sort_method='desc', 
                           average='median', min_data_months=0,
                           prob_frac=0.5,
                           bins_frac=0.4,
                           splice_frac=1.0,
                           apply_market_cap_filter=False,
                           exclude_stocks=None,
                           factors=None):

    if factors is None:
        factor_indexes = list(range(len(project['DataInfo']['Factors'])))
    else:
        factor_indexes = [project['DataInfo']['Factors'].index(f) for f in factors]

    factor_weights = {}

    if apply_market_cap_filter:
        base_returns_df = apply_market_cap_filter_to_returns(project).T
    else:
        base_returns_df = project['Data']['Returns'].copy()
    base_returns_df = returns_df.loc[data_start_date:end_date].T

    if exclude_stocks is not None:
        base_returns_df.loc[base_returns_df.index.isin(exclude_stocks)] = np.nan

    for factor_index in factor_indexes:

        factor_df = project['Data']['Factors'][factor_index].copy()
        factor_df = factor_df.loc[start_date:end_date].T
    
        returns_df = base_returns_df.copy()
        returns_df += 0*factor_df
        returns_df = returns_df.dropna(how='all', axis=1)

        factor_df += 0*returns_df
        factor_df = factor_df.dropna(how='all', axis=1)

        returns_pct_rank_df = (returns_df.rank(pct=False, axis=0, ascending=True)-1.0)
        returns_pct_rank_df /= returns_pct_rank_df.max()

        if False:
            returns_pct_rank_df -= 0.5
            returns_pct_rank_df *= returns_pct_rank_df.count()
            returns_pct_rank_df /= len(returns_pct_rank_df.index)
            returns_pct_rank_df += 0.5
            
        if method == 'corr':

            ascending = (sort_method == 'asc')

            if sort_method == 'asc':
                sorted_returns_pct_rank_df = factor_calc.sort_returns_by_factor(
                    factor_df, returns_pct_rank_df, ascending=True).dropna(how='all')
            elif sort_method == 'desc':
                sorted_returns_pct_rank_df = factor_calc.sort_returns_by_factor(
                    factor_df, returns_pct_rank_df, ascending=False).dropna(how='all')
            else:
                assert(sort_method == 'spliced')

                sorted_returns_pct_rank_df = factor_calc.sort_returns_by_factor(
                    factor_df, returns_pct_rank_df, ascending=True).dropna(how='all')

                sorted_returns_pct_rank_desc_df = factor_calc.sort_returns_by_factor(
                    factor_df, returns_pct_rank_df, ascending=False).dropna(how='all')

                sorted_returns_pct_rank_desc_df.index = sorted_returns_pct_rank_desc_df.index[::-1]

                sorted_returns_pct_rank_df.loc[sorted_returns_pct_rank_df.index > len(sorted_returns_pct_rank_df)/2] = sorted_returns_pct_rank_desc_df

                if splice_frac < 1.0:
                    assert(splice_frac > 0.0)
                    splice_min = int(0.5*splice_frac*len(sorted_returns_pct_rank_df.index))
                    splice_max = len(sorted_returns_pct_rank_df.index) - splice_min - 1
                    sorted_returns_pct_rank_df = sorted_returns_pct_rank_df.loc[
                        (sorted_returns_pct_rank_df.index < splice_min) | 
                        (sorted_returns_pct_rank_df.index > splice_max)
                    ].reset_index(drop=True)

            if min_data_months > 0:
                data_months = sorted_returns_pct_rank_df.count(axis=1)
                sorted_returns_pct_rank_df = sorted_returns_pct_rank_df.loc[data_months >= min_data_months]

            if average == 'median':
                average_sorted_returns_pct_rank = sorted_returns_pct_rank_df.median(axis=1)
            else:
                assert(average == 'mean')
                average_sorted_returns_pct_rank = sorted_returns_pct_rank_df.mean(axis=1)
            average_sorted_returns_pct_rank = average_sorted_returns_pct_rank.dropna()

            corr = np.corrcoef(average_sorted_returns_pct_rank, average_sorted_returns_pct_rank.index)[0][1]

            if sort_method == 'desc':
                corr = -corr

            factor_weights[factor_index] = corr
            
        elif method == 'prob':

            factor_pct_rank_df = (factor_df.rank(pct=False, axis=0, ascending=True)-1.0)
            factor_pct_rank_df /= factor_pct_rank_df.max()

            df = pd.DataFrame(data={ 
                    'f': factor_pct_rank_df.values.flatten(), 
                    'r': returns_pct_rank_df.values.flatten() 
                })

            p = max(prob_frac, df.f.min(), 1.0 - df.f.max())
            
            t_value, p_value = scipy.stats.ttest_ind(df.loc[df.f >= (1.0-p)].r, df.loc[df.f <= p].r)
            
            #factor_weights[factor_index] = np.sign(t_value) * (1.0 - p_value)
            factor_weights[factor_index] = t_value  
            
        elif method == 'binned_corr':

            factor_pct_rank_df = (factor_df.rank(pct=False, axis=0, ascending=True)-1.0)
            factor_pct_rank_df /= factor_pct_rank_df.max()

            bins = int(len(factor_pct_rank_df)*bins_frac)

            df = pd.DataFrame(data={ 
                    'f':  factor_pct_rank_df.values.flatten(), 
                    'r': returns_pct_rank_df.values.flatten() 
                })

            df['bin'] = (df.f * (bins-1)).round()

            binned_return_ranks = df.groupby('bin').r

            if average == 'median':            
                binned_return_ranks = binned_return_ranks.median()
            else:
                assert(average == 'mean')
                binned_return_ranks = binned_return_ranks.mean()

            corr = np.corrcoef(binned_return_ranks.index, binned_return_ranks.values)[0][1]

            factor_weights[factor_index] = corr

        else:
            assert(False)

    factor_weights = pd.Series(factor_weights)
    factor_weights.index.name = 'Factor'

    return factor_weights

def calc_rolling_factor_weightings(project, 
                                   data_start_date, 
                                   output_start_date, 
                                   rolling_months,
                                   end_date,
                                   method,
                                   sort_method='desc',
                                   average='median',
                                   min_data_months=0,
                                   prob_frac=0.5,
                                   bins_frac=0.4,
                                   splice_frac=1.0,
                                   apply_market_cap_filter=False,
                                   exclude_stocks=None,
                                   factors=None):
    
    end_date = pd.Timestamp(end_date)
    data_start_date = pd.Timestamp(data_start_date)
    output_start_date = pd.Timestamp(output_start_date)

    if method == 'monthly_corr':

        if apply_market_cap_filter:
            returns_df = apply_market_cap_filter_to_returns(project).T
        else:
            returns_df = project['Data']['Returns'].copy()
        returns_df = returns_df.loc[data_start_date:end_date]

        if exclude_stocks is not None:
            returns_df = returns_df.T
            returns_df.loc[returns_df.index.isin(exclude_stocks)] = np.nan
            returns_df = returns_df.T

        if factors is None:
            factor_indexes = list(range(len(project['DataInfo']['Factors'])))
        else:
            factor_indexes = [project['DataInfo']['Factors'].index(f) for f in factors]

        factor_weights = {}

        for factor_index in factor_indexes:

            factor_df = project['Data']['Factors'][factor_index].copy().loc[data_start_date:end_date]
            factor_df += 0*returns_df

            corrs = 100.0 * returns_df.corrwith(factor_df, axis=1, method='spearman')

            rolling = corrs.rolling(window=rolling_months, min_periods=min(12, rolling_months))
            if average == 'median':
                weights = rolling.median()
            elif average == 'mean':
                weights = rolling.mean()
            else:
                raise ValueError("Invalid Correlation average method '{}'".format(average))

            factor_weights[factor_index] = weights

        return pd.DataFrame(factor_weights).loc[output_start_date:end_date]


    factor_weights_by_month = {}

    month = pd.Timestamp(output_start_date)
    while True:

        calc_start_date = (month - pd.DateOffset(months=rolling_months-1)) + pd.offsets.MonthEnd(0)
        calc_start_date = max(calc_start_date, data_start_date)
        
        factor_weights_by_month[month] = calc_factor_weightings(project, calc_start_date, month, 
                                                                method,
                                                                sort_method=sort_method, 
                                                                average=average,
                                                                min_data_months=min_data_months,
                                                                prob_frac=prob_frac,
                                                                bins_frac=bins_frac,
                                                                splice_frac=splice_frac,
                                                                apply_market_cap_filter=apply_market_cap_filter,
                                                                exclude_stocks=exclude_stocks,
                                                                factors=factors)

        month = (month + pd.DateOffset(months=1)) + pd.offsets.MonthEnd(0)
        if month > end_date:
            break
            
        if month.month == 1:
            print(month)

    result_df = pd.DataFrame(factor_weights_by_month).T
    result_df.index.name = 'Date'
    return result_df


