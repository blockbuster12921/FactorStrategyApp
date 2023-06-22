import numpy as np

def calc_score(metrics_df):
    score = 100.0 * np.nan_to_num(metrics_df['Mean'].mean())
    score += 2.0 * np.nan_to_num(metrics_df['FractionPositive'].mean())
    score += 100.0 * np.nan_to_num(metrics_df['MeanNegative'].mean())
    return score

