import pandas as pd
import numpy as np
import logging
from typing import List
from .trading_solution import calc_sharpe
from .utils import MeasureTime, MeasureBlockTime

logger = logging.getLogger(__name__)

# Combine factors by taking the mean, ignoring any missing values
def combine_factors(factors_data, factors):

    assert(len(set(factors)) == len(factors))
    assert(len(factors) > 0)

    x = []
    for f in factors:
        if f < 0:
            x.append(1.0 - factors_data[-f-1].values)
        else:
            x.append(factors_data[f].values)
    x = np.array(x)

    f0 = factors[0]
    if f0 < 0:
        f0 = -f0 - 1
    f0 = factors_data[f0]
    factor_av = pd.DataFrame(index=f0.index, columns=f0.columns, data=np.nanmean(x, axis=0))

    return factor_av

# Combine factors by taking the weighted mean, ignoring any missing values
def combine_factors_weighted(factors_data, factors, weights):

    assert(len(set(factors)) == len(factors))
    assert(len(factors) > 0)
    assert(len(weights) == len(factors))

    if all(weights[0] == w for w in weights):
        return combine_factors(factors_data, factors)

    x = []
    w = []
    for i, f in enumerate(factors):
        if f < 0:
            x.append(weights[i]*(1.0 - factors_data[-f-1].values))
            w.append(weights[i] + 0*factors_data[-f-1].values)
        else:
            x.append(weights[i]*factors_data[f].values)
            w.append(weights[i] + 0*factors_data[f].values)
    x = np.array(x)
    w = np.array(w)
    
    y = np.nansum(x, axis=0) / np.nansum(w, axis=0) 

    f0 = factors[0]
    if f0 < 0:
        f0 = -f0 - 1
    f0 = factors_data[f0]
    factor_av = pd.DataFrame(index=f0.index, columns=f0.columns, data=y)

    return factor_av


# Calculate differences between returns for top N long-short pairs ranked by factor
def calc_long_short_pair_return_deltas(factors: pd.DataFrame, returns: pd.DataFrame, min_pairs=1, max_pairs=15) -> pd.DataFrame:

    assert(max_pairs >= min_pairs)

    long_short_returns_delta = pd.DataFrame(index=range(1,1+len(factors.index)), columns=factors.columns, data=0.0)

    # For each month
    for month in factors.columns:

        # Calculate returns sorted by factor descending (long) minus returns sorted by factor ascending (short)

        f = factors[month].values
        r = returns[month].values

        sorted_indexes = np.argsort(f, kind='stable')

        nan_count = np.sum(np.isnan(f))

        delta = r[sorted_indexes]
        if nan_count > 0:
            delta = delta[:-nan_count]

        delta = delta[::-1] - delta

        if nan_count > 0:
            long_short_returns_delta[month].iloc[:-nan_count] = delta
        else:
            long_short_returns_delta[month] = delta

    # Calculate expanding mean of N long-short deltas
    long_short_returns_delta = long_short_returns_delta.expanding(min_periods=1).mean().dropna(axis=0, how='all')

    long_short_returns_delta = long_short_returns_delta.loc[min_pairs:max_pairs]

    long_short_returns_delta.index.name = "Long-Short Pairs"
    return long_short_returns_delta.T

# Calculate metrics for a set of long-short pair deltas
def calc_long_short_pairs_metrics(long_short_pair_return_deltas: pd.DataFrame, start_date=None, end_date=None) -> pd.DataFrame:

    x = long_short_pair_return_deltas.loc[start_date:end_date].values
    
    count = np.nansum(1+x*0, axis=0)
    #count = x.shape[0] - np.isnan(x).sum()
    
    sum_ = np.nansum(x, axis=0)
    
    mean = sum_/count

    frac_positive = np.nanmean(x >= 0, axis=0)

    sum_negative = np.nansum(x*(x < 0), axis=0)

    mean_negative = sum_negative / count

    count_negative = np.nansum(x < 0, axis=0)
    
    sharpe = np.sqrt(12.0) * mean / np.nanstd(x, ddof=1, axis=0)

    metrics = pd.DataFrame(index=long_short_pair_return_deltas.columns, 
                           data={
                               'Count': count,
                               'Sum':sum_,
                               'Mean':mean,
                               'FractionPositive': frac_positive,
                               'SumNegative': sum_negative,
                               'MeanNegative': mean_negative,
                               'CountNegative': count_negative,
                               'Sharpe': sharpe,
                           })

    return metrics

# Return indexes sorted by factor value, excluding nans 
def calc_factor_ranking(factors: pd.Series) -> List[int]:

    assert(isinstance(factors, pd.Series))

    sorted_indexes = np.argsort(-factors.values, kind='stable')
    nan_count = np.sum(np.isnan(factors.values))
    if nan_count == 0:
        return factors.index.values[sorted_indexes].tolist()
    else:
        return factors.index.values[sorted_indexes][:-nan_count].tolist()

# Sort returns by corresponding factor values
def sort_returns_by_factor(factor_df, returns_df, ascending=True):

    assert(len(factor_df.index) == len(returns_df.index))

    sorted_returns = pd.DataFrame(index=range(1,1+len(factor_df.index)), columns=factor_df.columns, data=np.nan)

    for month in factor_df.columns:

        f = factor_df[month].values
        r = returns_df[month].values

        sorted_indexes = np.argsort(f, kind='stable')

        nan_count = np.sum(np.isnan(f))

        sorted_r = r[sorted_indexes]
        if nan_count > 0:
            sorted_r = sorted_r[:-nan_count]

        if not ascending:
            sorted_r = sorted_r[::-1]

        if nan_count > 0:
            sorted_returns[month].iloc[:-nan_count] = sorted_r
        else:
            sorted_returns[month] = sorted_r
            
    return sorted_returns

