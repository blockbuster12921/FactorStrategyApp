"use strict";

function initProjectSettings(project, factorStrategies) {

    const projectId = project['ID'];

    $('.project-settings-oos-container').prop('hidden', project['Status'] !== 'Test');
    $('#project-status-select').on('change', function () {
        $('.project-settings-oos-container').prop('hidden', $('#project-status-select').val() !== 'Test');
    });

    populateMonthSelectpicker($('#project-settings-oos-start-month-select'), null);
    populateYearSelectpicker($('#project-settings-oos-start-year-select'), 2000, moment().year(), null);
    if ( project['OOSStartDate'] !== null ) {
        const startDate = moment(project['OOSStartDate']);
        $('#project-settings-oos-start-month-select').val(startDate.month()+1).selectpicker('refresh');
        $('#project-settings-oos-start-year-select').val(startDate.year()).selectpicker('refresh');
    }
    $('#project-settings-oos-start-month-select').on('change', function () {
        updateProjectMetadata(projectId);
    });
    $('#project-settings-oos-start-year-select').on('change', function () {
        updateProjectMetadata(projectId);
    });

    populateMonthSelectpicker($('#project-settings-oos-end-month-select'), null);
    populateYearSelectpicker($('#project-settings-oos-end-year-select'), 2000, moment().year(), null);
    if ( project['OOSEndDate'] !== null ) {
        const endDate = moment(project['OOSEndDate']);
        $('#project-settings-oos-end-month-select').val(endDate.month()+1).selectpicker('refresh');
        $('#project-settings-oos-end-year-select').val(endDate.year()).selectpicker('refresh');
    }
    $('#project-settings-oos-end-month-select').on('change', function () {
        updateProjectMetadata(projectId);
    });
    $('#project-settings-oos-end-year-select').on('change', function () {
        updateProjectMetadata(projectId);
    });

    $('#project-settings-long-short-pairs-target').on('change', function(){
        updateProjectSetting(projectId, 'LongShortPairsTarget', Number($(this).val()));
    });
    $('#project-settings-long-short-pairs-delta').on('change', function(){
        updateProjectSetting(projectId, 'LongShortPairsDelta', Number($(this).val()));
    });

    $('#project-settings-factor-data-completeness-input').on('change', function(){
        updateProjectSetting(projectId, 'FactorDataCompletenessPercentage', Number($(this).val()));
    });

    $('#project-settings-market-cap-filter-input').on('change', function(){
        updateProjectSetting(projectId, 'MarketCapFilterValue', Number($(this).val()));
    });

    $('#project-settings-market-cap-filter-apply-select').on('change', function(){
        updateProjectSetting(projectId, 'MarketCapFilterStages', $(this).val());
    });

    const currentYear = moment().year();
    populateMonthSelectpicker($('#project-settings-market-cap-filter-month-select'), null);
    populateYearSelectpicker($('#project-settings-market-cap-filter-year-select'), 2000, currentYear, null);
    $('.project-settings-market-cap-filter-date-select').on('change', function(){
        const date = moment().set({
            'year': parseInt($('#project-settings-market-cap-filter-year-select').val()), 
            'month': parseInt($('#project-settings-market-cap-filter-month-select').val())-1,
        }).endOf('month').format("Y-MM-DD");
        updateProjectSetting(projectId, 'MarketCapFilterDate', date);
    });

    refreshProjectFactorsInfo(projectId);
    $('#project-settings-factors-enabled-select').on('change', function(){
        updateFactorsDisabledStatus(projectId);
    });

    $('#project-settings-returns-outlier-rejection-method-select').on('change', function(){
        updateProjectSetting(projectId, 'ReturnsMonthlyOutlierRejectionMethod', $(this).val());
        $('.project-settings-returns-outlier-rejection-setting').prop('hidden', $(this).val() === "none");
    });
    $('#project-settings-returns-outlier-rejection-cutoff-input').on('change', function(){
        updateProjectSetting(projectId, 'ReturnsMonthlyOutlierRejectionCutoff', Number($(this).val()));
    });

    $('#project-settings-average-returns-outlier-range-start-input').on('change', function(){
        updateProjectSetting(projectId, 'AverageReturnsOutlierRangeStart', Number($(this).val()));
    });
    $('#project-settings-average-returns-outlier-range-end-input').on('change', function(){
        updateProjectSetting(projectId, 'AverageReturnsOutlierRangeEnd', Number($(this).val()));
    });

    $('#project-settings-driver-method-select').on('change', function(){
        updateProjectSetting(projectId, 'DriverMethod', $(this).val());
        $('.project-settings-driver-monthly-correlation-setting').prop('hidden', $(this).val() !== "correlation");
        $('.project-settings-driver-return-rank-correlation-setting').prop('hidden', $(this).val() !== "rank_average_correlation");
        $('.project-settings-driver-return-rank-ttest-setting').prop('hidden', $(this).val() !== "rank_ttest");
        $('.project-settings-driver-binned-return-rank-correlation-setting').prop('hidden', $(this).val() !== "binned_rank_correlation");
        $('.project-settings-driver-average-by-settings').prop('hidden', $(this).val() === "rank_ttest");
    });

    populateMonthSelectpicker($('#project-settings-driver-insample-start-month-select'), null);
    populateYearSelectpicker($('#project-settings-driver-insample-start-year-select'), 1990, currentYear, null);
    $('.project-settings-driver-insample-start-select').on('change', function(){
        const date = moment().set({
            'year': parseInt($('#project-settings-driver-insample-start-year-select').val()), 
            'month': parseInt($('#project-settings-driver-insample-start-month-select').val())-1,
        }).endOf('month').format("Y-MM-DD");
        updateProjectSetting(projectId, 'DriverInSampleStartDate', date);
    });

    populateMonthSelectpicker($('#project-settings-driver-insample-end-month-select'), null);
    populateYearSelectpicker($('#project-settings-driver-insample-end-year-select'), 1990, currentYear, null);
    $('.project-settings-driver-insample-end-select').on('change', function(){
        const date = moment().set({
            'year': parseInt($('#project-settings-driver-insample-end-year-select').val()), 
            'month': parseInt($('#project-settings-driver-insample-end-month-select').val())-1,
        }).endOf('month').format("Y-MM-DD");
        updateProjectSetting(projectId, 'DriverInSampleEndDate', date);
    });

    $('#project-settings-factor-outlier-rejection-input').on('change', function(){
        updateProjectSetting(projectId, 'FactorOutlierRejection', Number($(this).val()));
    });
    $('#project-settings-factor-outlier-rejection-range-start-input').on('change', function(){
        updateProjectSetting(projectId, 'FactorOutlierRejectionRangeStart', Number($(this).val()));
    });
    $('#project-settings-factor-outlier-rejection-range-end-input').on('change', function(){
        updateProjectSetting(projectId, 'FactorOutlierRejectionRangeEnd', Number($(this).val()));
    });
    $('#project-settings-factor-outlier-rejection-range-step-input').on('change', function(){
        updateProjectSetting(projectId, 'FactorOutlierRejectionRangeStep', Number($(this).val()));
    });

    $('#project-settings-driver-returns-outlier-rejection-default-input').on('change', function(){
        updateProjectSetting(projectId, 'DriverReturnsOutlierRejectionDefault', Number($(this).val()));
    });
    $('#project-settings-driver-returns-outlier-rejection-range-start-input').on('change', function(){
        updateProjectSetting(projectId, 'DriverReturnsOutlierRejectionRangeStart', Number($(this).val()));
    });
    $('#project-settings-driver-returns-outlier-rejection-range-end-input').on('change', function(){
        updateProjectSetting(projectId, 'DriverReturnsOutlierRejectionRangeEnd', Number($(this).val()));
    });
    $('#project-settings-driver-returns-outlier-rejection-range-step-input').on('change', function(){
        updateProjectSetting(projectId, 'DriverReturnsOutlierRejectionRangeStep', Number($(this).val()));
    });
    $('#project-settings-driver-correlation-average-method-select').on('change', function(){
        updateProjectSetting(projectId, 'DriverCorrelationAverageMethod', $(this).val());
    });
    $('#project-settings-driver-correlation-average-months-input').on('change', function(){
        updateProjectSetting(projectId, 'DriverCorrelationAverageMonths', Number($(this).val()));
    });
    $('#project-settings-driver-correlation-basis-select').on('change', function(){
        updateProjectSetting(projectId, 'DriverCorrelationBasis', $(this).val());
    });
    $('#project-settings-driver-factor-weight-scale-input').on('change', function(){
        updateProjectSetting(projectId, 'DriverFactorWeightScale', Number($(this).val()));
    });
    $('#project-settings-driver-top-factors-input').on('change', function(){
        updateProjectSetting(projectId, 'DriverTopFactors', Number($(this).val()));
    });

    $('#project-settings-driver-monthly-correlation-min-data-months').on('change', function(){
        updateProjectSetting(projectId, 'DriverCorrelationMinMonths', Number($(this).val()));
    });

    $('#project-settings-driver-return-rank-correlation-order-select').on('change', function(){
        updateProjectSetting(projectId, 'DriverReturnRankCorrelationOrder', $(this).val());
        $('.project-settings-driver-return-rank-correlation-order-spliced-setting').prop('hidden', $(this).val() !== 'spliced');
    });
    $('#project-settings-driver-return-rank-correlation-splice-percent-input').on('change', function(){
        updateProjectSetting(projectId, 'DriverReturnRankCorrelationSplicePercent', Number($(this).val()));
    });
    $('#project-settings-driver-return-rank-correlation-min-data-months').on('change', function(){
        updateProjectSetting(projectId, 'DriverReturnRankCorrelationMinDataMonths', Number($(this).val()));
    });

    $('#project-settings-driver-return-rank-ttest-percentage-input').on('change', function(){
        updateProjectSetting(projectId, 'DriverReturnRankTTestPercentage', Number($(this).val()));
    });

    $('#project-settings-driver-binned-return-rank-correlation-bin-percentage-input').on('change', function(){
        updateProjectSetting(projectId, 'DriverBinnedReturnRankCorrelationBinPercentage', Number($(this).val()));
    });

    $('#project-settings-ticker-selection-min-stocks-input').on('change', function(){
        updateProjectSetting(projectId, 'TickerSelectionMinStocks', Number($(this).val()));
    });
    $('#project-settings-ticker-selection-score-impact-tolerance-input').on('change', function(){
        updateProjectSetting(projectId, 'TickerSelectionScoreImpactTolerance', Number($(this).val()));
    });

    let html = '';
    factorStrategies.forEach(function(strategy) {
        html += '<option value="' + strategy['ID'] + '">';
        html += 'Combinations: ' + strategy['Description']['combinations'] + '; ';
        html += 'Factors: ' + strategy['Description']['factors'] + '; ';
        html += strategy['Description']['weighting'] + ' Weighting';
        html += '</option>';
    });
    $('#project-settings-factor-selection-strategy-select').html(html).selectpicker('refresh');
    $('#project-settings-factor-selection-strategy-select').on('change', function(){
        updateProjectSetting(projectId, 'FactorSelectionStrategies', $(this).val());
    });

    $('#project-settings-factor-selection-period-duration-input').on('change', function(){
        updateProjectSetting(projectId, 'FactorSelectionPeriodDuration', Number($(this).val()));
    });
    $('#project-settings-factor-selection-st-period-duration-input').on('change', function(){
        updateProjectSetting(projectId, 'FactorSelectionSTPeriodDuration', Number($(this).val()));
    });
    $('#project-settings-factor-selection-st-period-weight-input').on('change', function(){
        updateProjectSetting(projectId, 'FactorSelectionSTPeriodWeight', Number($(this).val()));
    });
    $('#project-settings-factor-selection-objective-select').on('change', function(){
        updateProjectSetting(projectId, 'FactorSelectionObjective', $(this).val());
        $('.project-settings-factor-selection-long-short-pairs-settings').prop('hidden', $(this).val() !== "score");
    });

    $('#project-settings-factor-selection-long-short-pairs-start-input').on('change', function(){
        updateProjectSetting(projectId, 'FactorSelectionLongShortPairsStart', Number($(this).val()));
    });
    $('#project-settings-factor-selection-long-short-pairs-end-input').on('change', function(){
        updateProjectSetting(projectId, 'FactorSelectionLongShortPairsEnd', Number($(this).val()));
    });
    $('#project-settings-factor-selection-combinations-from-select').on('change', function(){
        updateProjectSetting(projectId, 'FactorSelectionCombinationGenerationMethods', $(this).val());
    });

    $('#project-settings-factor-clustering-distance-threshold-scale-input').on('change', function(){
        updateProjectSetting(projectId, 'FactorClusteringDistanceThresholdMultiplier', Number($(this).val()));
    });
    $('#project-settings-factor-clustering-rolling-duration-input').on('change', function(){
        updateProjectSetting(projectId, 'FactorClusteringDurationMonths', Number($(this).val()));
    });
    populateMonthSelectpicker($('#project-settings-factor-clustering-start-month-select'), null);
    populateYearSelectpicker($('#project-settings-factor-clustering-start-year-select'), 1990, currentYear, null);
    $('.project-settings-factor-clustering-start-select').on('change', function(){
        const date = moment().set({
            'year': parseInt($('#project-settings-factor-clustering-start-year-select').val()), 
            'month': parseInt($('#project-settings-factor-clustering-start-month-select').val())-1,
        }).endOf('month').format("Y-MM-DD");
        updateProjectSetting(projectId, 'FactorClusteringStartDate', date);
    });

    $('#project-settings-factor-selection-filter-min-factors-input').on('change', function(){
        updateProjectSetting(projectId, 'FactorFilterMinFactorCount', Number($(this).val()));
    });
    $('#project-settings-factor-selection-filter-removal-percentage-input').on('change', function(){
        updateProjectSetting(projectId, 'FactorFilterRemovalPercentage', Number($(this).val()));
    });
    $('#project-settings-factor-selection-optimize-factor-count-lo-input').on('change', function(){
        updateProjectSetting(projectId, 'FactorSelectionOptimizeFactorCountLo', Number($(this).val()));
    });
    $('#project-settings-factor-selection-optimize-factor-count-hi-input').on('change', function(){
        updateProjectSetting(projectId, 'FactorSelectionOptimizeFactorCountHi', Number($(this).val()));
    });
    $('#project-settings-factor-selection-optimize-min-combinations-input').on('change', function(){
        updateProjectSetting(projectId, 'FactorSelectionOptimizeMinCombinations', Number($(this).val()));
    });
    $('#project-settings-factor-selection-generate-factor-count-lo-input').on('change', function(){
        updateProjectSetting(projectId, 'FactorSelectionGenerateFactorCountLo', Number($(this).val()));
    });
    $('#project-settings-factor-selection-generate-factor-count-hi-input').on('change', function(){
        updateProjectSetting(projectId, 'FactorSelectionGenerateFactorCountHi', Number($(this).val()));
    });
    $('#project-settings-factor-selection-generate-max-combinations-per-stage-input').on('change', function(){
        updateProjectSetting(projectId, 'FactorSelectionGenerateMaxCombinationsPerStage', Number($(this).val()));
    });
    $('#project-settings-factor-selection-generate-top-factors-count-input').on('change', function(){
        updateProjectSetting(projectId, 'FactorSelectionGenerateTopFactorCount', Number($(this).val()));
    });
    $('#project-settings-factor-selection-generate-top-factors-objective-select').on('change', function(){
        updateProjectSetting(projectId, 'FactorSelectionGenerateTopFactorObjective', $(this).val());
    });

    $('#project-reset-settings-button').on('click', function(){
        resetProjectSettings(projectId);
        resetProjectFactorsDisabledStatus(projectId);
    });

    refreshProjectSettings(projectId);
    refreshProjectFactorsInfo(projectId);
}

function refreshProjectSettings(projectId) {

    var promise = $.ajax({
        url: '/get_project_settings',
        type: 'POST',
        data: JSON.stringify({ 'ProjectID': projectId }),
        contentType: "application/json; charset=utf-8",
        dataType: "text",
        success: null,
    });

    promise.fail(function( jqXHR, textStatus ) {
        console.log("get_project_settings failed");
        console.log(textStatus);
        return;
    });

    promise.done(function (response) {
        console.log("get_project_settings succeeded");

        try {
            var settings = JSON.parse(response);
        } 
        catch (e) {
            console.log("Failed to parse response JSON");
            return;
        };

        function updateSettingInput(settings, key, elementId) {
            if ( key in settings ) {
                $('#'+elementId).val(settings[key]);
            }
        }
        function updateSettingSelect(settings, key, elementId) {
            if ( key in settings ) {
                $('#'+elementId).val(settings[key]).selectpicker('refresh');
            }
        }
        function updateSettingDateSelects(settings, key, monthElementId, yearElementId) {
            if ( key in settings ) {
                const date = moment(settings[key]);
                $('#'+monthElementId).val(date.month()+1).selectpicker('refresh');
                $('#'+yearElementId).val(date.year()).selectpicker('refresh');
            }
        }
    
        updateSettingInput(settings, 'LongShortPairsTarget', 'project-settings-long-short-pairs-target');
        updateSettingInput(settings, 'LongShortPairsDelta', 'project-settings-long-short-pairs-delta');
        updateSettingInput(settings, 'FactorDataCompletenessPercentage', 'project-settings-factor-data-completeness-input');
        updateSettingInput(settings, 'MarketCapFilterValue', 'project-settings-market-cap-filter-input');
        updateSettingDateSelects(settings, 'MarketCapFilterDate', 'project-settings-market-cap-filter-month-select', 'project-settings-market-cap-filter-year-select');
        updateSettingSelect(settings, 'MarketCapFilterStages', 'project-settings-market-cap-filter-apply-select');
        updateSettingSelect(settings, 'ReturnsMonthlyOutlierRejectionMethod', 'project-settings-returns-outlier-rejection-method-select');
        $('.project-settings-returns-outlier-rejection-setting').prop('hidden', $('#project-settings-returns-outlier-rejection-method-select').val() === "none");
        updateSettingInput(settings, 'ReturnsMonthlyOutlierRejectionCutoff', 'project-settings-returns-outlier-rejection-cutoff-input');
        updateSettingInput(settings, 'AverageReturnsOutlierRangeStart', 'project-settings-average-returns-outlier-range-start-input');
        updateSettingInput(settings, 'AverageReturnsOutlierRangeEnd', 'project-settings-average-returns-outlier-range-end-input');
        updateSettingSelect(settings, 'DriverMethod', 'project-settings-driver-method-select');
        $('.project-settings-driver-monthly-correlation-setting').prop('hidden', $('#project-settings-driver-method-select').val() !== "correlation");
        $('.project-settings-driver-return-rank-correlation-setting').prop('hidden', $('#project-settings-driver-method-select').val() !== "rank_average_correlation");
        $('.project-settings-driver-return-rank-ttest-setting').prop('hidden', $('#project-settings-driver-method-select').val() !== "rank_ttest");
        $('.project-settings-driver-binned-return-rank-correlation-setting').prop('hidden', $('#project-settings-driver-method-select').val() !== "binned_rank_correlation");
        $('.project-settings-driver-average-by-settings').prop('hidden', $('#project-settings-driver-method-select').val() === "rank_ttest");
        updateSettingDateSelects(settings, 'DriverInSampleStartDate', 'project-settings-driver-insample-start-month-select', 'project-settings-driver-insample-start-year-select');
        updateSettingDateSelects(settings, 'DriverInSampleEndDate', 'project-settings-driver-insample-end-month-select', 'project-settings-driver-insample-end-year-select');
        updateSettingInput(settings, 'FactorOutlierRejection', 'project-settings-factor-outlier-rejection-input');
        updateSettingInput(settings, 'FactorOutlierRejectionRangeStart', 'project-settings-factor-outlier-rejection-range-start-input');
        updateSettingInput(settings, 'FactorOutlierRejectionRangeEnd', 'project-settings-factor-outlier-rejection-range-end-input');
        updateSettingInput(settings, 'FactorOutlierRejectionRangeStep', 'project-settings-factor-outlier-rejection-range-step-input');
        updateSettingInput(settings, 'DriverReturnsOutlierRejectionDefault', 'project-settings-driver-returns-outlier-rejection-default-input');
        updateSettingInput(settings, 'DriverReturnsOutlierRejectionRangeStart', 'project-settings-driver-returns-outlier-rejection-range-start-input');
        updateSettingInput(settings, 'DriverReturnsOutlierRejectionRangeEnd', 'project-settings-driver-returns-outlier-rejection-range-end-input');
        updateSettingInput(settings, 'DriverReturnsOutlierRejectionRangeStep', 'project-settings-driver-returns-outlier-rejection-range-step-input');
        updateSettingSelect(settings, 'DriverCorrelationAverageMethod', 'project-settings-driver-correlation-average-method-select');
        updateSettingInput(settings, 'DriverCorrelationAverageMonths', 'project-settings-driver-correlation-average-months-input');
        updateSettingSelect(settings, 'DriverCorrelationBasis', 'project-settings-driver-correlation-basis-select');
        updateSettingSelect(settings, 'DriverFactorWeightScale', 'project-settings-driver-factor-weight-scale-input');
        updateSettingSelect(settings, 'DriverTopFactors', 'project-settings-driver-top-factors-input');
        updateSettingSelect(settings, 'DriverCorrelationMinMonths', 'project-settings-driver-monthly-correlation-min-data-months');    
        updateSettingSelect(settings, 'DriverReturnRankCorrelationOrder', 'project-settings-driver-return-rank-correlation-order-select');
        $('.project-settings-driver-return-rank-correlation-order-spliced-setting').prop('hidden', $('#project-settings-driver-return-rank-correlation-order-select').val() !== 'spliced');
        updateSettingInput(settings, 'DriverReturnRankCorrelationSplicePercent', 'project-settings-driver-return-rank-correlation-splice-percent-input');
        updateSettingInput(settings, 'DriverReturnRankCorrelationMinDataMonths', 'project-settings-driver-return-rank-correlation-min-data-months');
        updateSettingInput(settings, 'DriverReturnRankTTestPercentage', 'project-settings-driver-return-rank-ttest-percentage-input');
        updateSettingInput(settings, 'DriverBinnedReturnRankCorrelationBinPercentage', 'project-settings-driver-binned-return-rank-correlation-bin-percentage-input');
    
        updateSettingInput(settings, 'TickerSelectionMinStocks', 'project-settings-ticker-selection-min-stocks-input');
        updateSettingInput(settings, 'TickerSelectionScoreImpactTolerance', 'project-settings-ticker-selection-score-impact-tolerance-input');
        updateSettingSelect(settings, 'FactorSelectionStrategies', 'project-settings-factor-selection-strategy-select');
        updateSettingInput(settings, 'FactorSelectionPeriodDuration', 'project-settings-factor-selection-period-duration-input');
        updateSettingInput(settings, 'FactorSelectionSTPeriodDuration', 'project-settings-factor-selection-st-period-duration-input');
        updateSettingInput(settings, 'FactorSelectionSTPeriodWeight', 'project-settings-factor-selection-st-period-weight-input');
        updateSettingSelect(settings, 'FactorSelectionObjective', 'project-settings-factor-selection-objective-select');
        updateSettingInput(settings, 'FactorSelectionLongShortPairsStart', 'project-settings-factor-selection-long-short-pairs-start-input');
        updateSettingInput(settings, 'FactorSelectionLongShortPairsEnd', 'project-settings-factor-selection-long-short-pairs-end-input');
        $('.project-settings-factor-selection-long-short-pairs-settings').prop('hidden', $('#project-settings-factor-selection-objective-select').val() !== "score");
        updateSettingSelect(settings, 'FactorSelectionCombinationGenerationMethods', 'project-settings-factor-selection-combinations-from-select');
    
        updateSettingInput(settings, 'FactorClusteringDistanceThresholdMultiplier', 'project-settings-factor-clustering-distance-threshold-scale-input');
        updateSettingInput(settings, 'FactorClusteringDurationMonths', 'project-settings-factor-clustering-rolling-duration-input');
        updateSettingDateSelects(settings, 'FactorClusteringStartDate', 'project-settings-factor-clustering-start-month-select', 'project-settings-factor-clustering-start-year-select');
    
        updateSettingInput(settings, 'FactorFilterMinFactorCount', 'project-settings-factor-selection-filter-min-factors-input');
        updateSettingInput(settings, 'FactorFilterRemovalPercentage', 'project-settings-factor-selection-filter-removal-percentage-input');
        updateSettingInput(settings, 'FactorSelectionOptimizeFactorCountLo', 'project-settings-factor-selection-optimize-factor-count-lo-input');
        updateSettingInput(settings, 'FactorSelectionOptimizeFactorCountHi', 'project-settings-factor-selection-optimize-factor-count-hi-input');
        updateSettingInput(settings, 'FactorSelectionOptimizeMinCombinations', 'project-settings-factor-selection-optimize-min-combinations-input');
        updateSettingInput(settings, 'FactorSelectionGenerateFactorCountLo', 'project-settings-factor-selection-generate-factor-count-lo-input');
        updateSettingInput(settings, 'FactorSelectionGenerateFactorCountHi', 'project-settings-factor-selection-generate-factor-count-hi-input');
        updateSettingInput(settings, 'FactorSelectionGenerateMaxCombinationsPerStage', 'project-settings-factor-selection-generate-max-combinations-per-stage-input');
        updateSettingInput(settings, 'FactorSelectionGenerateTopFactorCount', 'project-settings-factor-selection-generate-top-factors-count-input');
        updateSettingSelect(settings, 'FactorSelectionGenerateTopFactorObjective', 'project-settings-factor-selection-generate-top-factors-objective-select');
    
    });
}

function updateProjectSetting(projectId, settingKey, settingValue) {

    let request = { 'ProjectID': projectId, 'Settings': { } };
    request['Settings'][settingKey] = settingValue;

    var promise = $.ajax({
        url: '/set_project_settings',
        type: 'POST',
        data: JSON.stringify(request),
        contentType: "application/json; charset=utf-8",
        dataType: "text",
        success: null,
    });

    promise.fail(function( jqXHR, textStatus ) {
        console.log("set_project_settings failed");
        console.log(textStatus);
        return;
    });

    promise.done(function (response) {
        console.log("set_project_settings succeeded");
    });
}

function refreshProjectFactorsInfo(projectId) {

    var promise = $.ajax({
        url: '/get_project_factors_info',
        type: 'POST',
        data: JSON.stringify({'ProjectID': projectId}),
        contentType: "application/json; charset=utf-8",
        dataType: "text",
        success: null,
    });

    promise.fail(function( jqXHR, textStatus ) {
        console.log("get_project_factors_info failed");
        console.log(textStatus);
        return;
    });

    promise.done(function (response) {
        console.log("get_project_factors_info succeeded");

        try {
            var responseJson = JSON.parse(response);
        } 
        catch (e) {
            console.log("Failed to parse response JSON");
            return;
        };

        if ( responseJson !== null ) {
            let html = '';
            let selected = [];
            responseJson['Factors'].forEach(function(factor){
                html += '<option value="' + factor['Name'] + '"';
                if ( factor['Enabled'] ) {
                    html += ' selected';
                    selected.push(factor['Name']);
                }
                html += '>' + factor['Name'] + '</option>';
            });
            $('#project-settings-factors-enabled-select').html(html).selectpicker('refresh');
            $('#project-settings-factors-enabled-select').val(selected).selectpicker('refresh');
        }
        else {
            $('#project-settings-factors-enabled-select').html('').selectpicker('refresh');
        }

    });

    return promise;
}

function updateFactorsDisabledStatus(projectId) {

    let disabledFactors = [];
    $('#project-settings-factors-enabled-select option:not(:selected)').each(function(){
        disabledFactors.push($(this).val());
    });

    let request = { 'ProjectID': projectId, 'FactorsDisabled': disabledFactors };

    var promise = $.ajax({
        url: '/set_project_factors_disabled_status',
        type: 'POST',
        data: JSON.stringify(request),
        contentType: "application/json; charset=utf-8",
        dataType: "text",
        success: null,
    });

    promise.fail(function( jqXHR, textStatus ) {
        console.log("set_project_factors_disabled_status failed");
        console.log(textStatus);
    });

    promise.done(function (response) {
        console.log("set_project_factors_disabled_status succeeded");
    });

    return promise;
}

function resetProjectSettings(projectId) {

    let request = { 'ProjectID': projectId, 'Settings': true, 'FactorsDisabled': true, 'StocksDisabled': false, };

    var promise = $.ajax({
        url: '/clear_project_settings',
        type: 'POST',
        data: JSON.stringify(request),
        contentType: "application/json; charset=utf-8",
        dataType: "text",
        success: null,
    });

    promise.fail(function( jqXHR, textStatus ) {
        console.log("clear_project_settings failed");
        console.log(textStatus);
    });

    promise.done(function (response) {
        console.log("clear_project_settings succeeded");
        refreshProjectSettings(projectId);
    });

    return promise;
}

function resetProjectFactorsDisabledStatus(projectId) {

    let request = { 'ProjectID': projectId, 'FactorsDisabled': [] };

    var promise = $.ajax({
        url: '/set_project_factors_disabled_status',
        type: 'POST',
        data: JSON.stringify(request),
        contentType: "application/json; charset=utf-8",
        dataType: "text",
        success: null,
    });

    promise.fail(function( jqXHR, textStatus ) {
        console.log("set_project_factors_disabled_status failed");
        console.log(textStatus);
    });

    promise.done(function (response) {
        console.log("set_project_factors_disabled_status succeeded");
        refreshProjectFactorsInfo(projectId);    
    });

    return promise;
}