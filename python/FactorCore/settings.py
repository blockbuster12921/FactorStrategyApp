import pandas as pd

default_project_settings = {
    'LongShortPairsTarget': 8,
    'LongShortPairsDelta': 2,
    'MarketCapFilterValue': 10000.0,
    'MarketCapFilterDate': '2020-01-31',
    'MarketCapFilterStages': ['FactorFilter','FactorOptimize','FactorGenerate','FactorStrategies'],
    'AverageReturnsOutlierRangeStart': -100.0,
    'AverageReturnsOutlierRangeEnd': 100.0,
    'ReturnsMonthlyOutlierRejectionMethod': 'none',
    'ReturnsMonthlyOutlierRejectionCutoff': 3.5,
    'DriverMethod': 'correlation',
    'DriverInSampleStartDate': '2004-12-31',
    'DriverInSampleEndDate': '2005-11-30',
    'FactorOutlierRejection': 100.0,
    'FactorOutlierRejectionRangeStart': 100.0,
    'FactorOutlierRejectionRangeEnd': 100.0,
    'FactorOutlierRejectionRangeStep': 0.5,
    'DriverReturnsOutlierRejectionDefault': 100.0,
    'DriverReturnsOutlierRejectionRangeStart': 100,
    'DriverReturnsOutlierRejectionRangeEnd': 100,
    'DriverReturnsOutlierRejectionRangeStep': 0.05,
    'DriverFactorWeightScale': 0,
    'DriverTopFactors': 1000,
    'DriverCorrelationAverageMethod': 'median',
    'DriverCorrelationAverageMonths': 120,
    'DriverCorrelationMinMonths': 12,
    'DriverCorrelationBasis': 'rolling_out_of_sample',
    'DriverReturnRankCorrelationOrder': 'spliced',
    'DriverReturnRankCorrelationSplicePercent': 70.0,
    'DriverReturnRankCorrelationMinDataMonths': 36,
    'DriverReturnRankTTestPercentage': 25,
    'DriverBinnedReturnRankCorrelationBinPercentage': 40,
    'TickerSelectionPeriodDuration': 36,
    'TickerSelectionMinStocks': 10000,
    'TickerSelectionScoreImpactTolerance': 0.1,
    'FactorSelectionStrategies': ['ce36be66-e4af-4208-81ed-7d43a9e853e5'],
    'FactorSelectionPeriodDuration': 12,
    'FactorSelectionSTPeriodDuration': 0,
    'FactorSelectionSTPeriodWeight': 0.0,
    'FactorSelectionObjective': 'rank_corr',
    'FactorSelectionLongShortPairsStart': 1,
    'FactorSelectionLongShortPairsEnd': 10,
    'FactorSelectionCombinationGenerationMethods': ['optimize'],
    'FactorSelectionOptimizeFactorCountLo': 5,
    'FactorSelectionOptimizeFactorCountHi': 25,
    'FactorSelectionOptimizeMinCombinations': 10000,
    'FactorSelectionGenerateFactorCountLo': 1,
    'FactorSelectionGenerateFactorCountHi': 10,
    'FactorSelectionGenerateMaxCombinationsPerStage': 20,
    'FactorSelectionGenerateTopFactorCount': 5,
    'FactorSelectionGenerateTopFactorObjective': 'rank_corr',
    'FactorFilterMinFactorCount': 500,
    'FactorFilterRemovalPercentage': 5.0,
    'FactorDataCompletenessPercentage': 70.0,
    'FactorClusteringDurationMonths': 120,
    'FactorClusteringStartDate': '2006-01-31',
    'FactorClusteringDistanceThresholdMultiplier': 0,
}

def overlay_default_project_settings(settings):
    if settings is None:
        return { **default_project_settings }
    else:
        return { **default_project_settings, **settings }

