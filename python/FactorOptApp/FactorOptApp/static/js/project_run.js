"use strict";

function initProjectRun(project, factorStrategies) {

    $('#start-project-run-button').on('click', function() { 
        $('#start-project-run-button').prop("disabled", true);
        $('#stop-project-run-button').prop("disabled", false);
        runProject(project);
    });

    $('#stop-project-run-button').on('click', function() { 
        $('#start-project-run-button').prop("disabled", false);
        $('#stop-project-run-button').prop("disabled", true);
    });

    $('#project-run-reset-button').on('click', function() { 
        resetProjectRun(project);
    });

    $('#project-excel-export-stock-selection-button').on('click', function() { 
        exportStockSelection(project, $(this).find(".button-progress-spinner"));
    });

    $('#project-excel-export-factors-disabled-button').on('click', function() { 
        exportFactorsDisabled(project, $(this).find(".button-progress-spinner"));
    });

    $('#project-excel-export-driver-param-selection-button').on('click', function() { 
        exportDriverParamSelection(project, $(this).find(".button-progress-spinner"));
    });

    $('#project-excel-export-factor-clusters-button').on('click', function() { 
        exportFactorClusters(project, $(this).find(".button-progress-spinner"));
    });

    $('#project-excel-export-driver-factor-weights-button').on('click', function() { 
        exportDriverFactorWeights(project, $(this).find(".button-progress-spinner"));
    });

    $('#project-excel-export-factor-filter-button').on('click', function() { 
        exportFactorFilter(project, $(this).find(".button-progress-spinner"));
    });

    $('#project-excel-export-combinations-button').on('click', function() { 
        exportCombinations(project, null, $(this).find(".button-progress-spinner"));
    });

    $('#project-excel-export-analysis-button').on('click', function() { 
        exportAnalysis(project, $(this).find(".button-progress-spinner"));
    });

    $('#stock-ranking-report-modal-submit-button').on('click', function() { 
        exportStockRanking(project, $('#project-excel-export-stock-ranking-button').find(".button-progress-spinner"));
    });

    let html = '';
    factorStrategies.forEach(function(strategy, index) {
        html += '<option value="' + strategy['ID'] + '">';
        html += 'Combinations: ' + strategy['Description']['combinations'] + '; ';
        html += 'Factors: ' + strategy['Description']['factors'] + '; ';
        html += strategy['Description']['weighting'] + ' Weighting';
        html += '</option>';
    });
    $('#stock-ranking-report-strategy-select').html(html).selectpicker('refresh');
    if ( factorStrategies.length > 0 ) {
        $('#stock-ranking-report-strategy-select').selectpicker('val', '8fe1a3e1-fb3f-416e-b5d3-6fae59587ee1');
    }


    initProjectRunHistoryTable();

    updateProjectRunState(project['ID']);
}

function refreshProjectRun(projectId) {
    $('#project-run-progress').prop('hidden', true);
    $('#project-run-complete').prop('hidden', true);
    $('#start-project-run-button').prop("disabled", false);
    $('#stop-project-run-button').prop("disabled", true);
    updateProjectRunState(projectId);
}

function runHistoryStageToText(data) {
    if ( data === null ) {
        return ""
    }
    if ( data === 'DisableStocks' ) {
        return 'Disable Stocks';
    }
    if ( data === 'DisableFactors' ) {
        return 'Disable Factors';
    }
    if ( data === 'TickerSelection' ) {
        return 'Ticker Selection';
    }
    if ( data === 'ReturnsOutlierRejectionSelection' ) {
        return 'Returns Outlier Rejection Selection'
    }
    if ( data === 'FactorOutlierRejectionSelection' ) {
        return 'Factor Outlier Rejection Selection'
    }
    if ( data === 'FactorOptimize' ) {
        return 'Optimize Factors';
    }
    if ( data === 'FactorGenerate' ) {
        return 'Generate Factors';
    }
    if ( data === 'FactorFilter' ) {
        return 'Filter Factors';
    }
    if ( data === 'FactorStrategies' ) {
        return 'Factor Strategies';
    }
    if ( data === 'FactorClustering' ) {
        return 'Factor Clustering';
    }
    return data;
}

function initProjectRunHistoryTable() {

    function dtRenderStage(data, type, row, meta) {
        return runHistoryStageToText(data);
    }

    function dtRenderStatus(data, type, row, meta) {
        if ( data === null ) {
            return ""
        }
        if ( data === 'InProgress' ) {
            return 'In Progress';
        }
        return data;
    }

    const columns = [
        { data: "Timestamp", title: "Time", className: "all", render: dtRenderTimestamp },
        { data: "Stage", title: "Stage", className: "all", render: dtRenderStage },  
        { data: "Status", title: "Status", className: "all", render: dtRenderStatus },  
        { data: "TargetMonth", title: "Target Month", className: "all", render: dtRenderMonth },
        { data: "Detail", title: "Detail", className: "all", defaultContent: "" },
    ]
    initDataTable('project-run-history-table', [], columns, "items", [[0, "desc"]], '');
}

function updateProjectRunState(projectId) {

    let request = { 'ProjectID': projectId };

    var promise = $.ajax({
        url: '/get_project_run_state',
        type: 'PUT',
        data: JSON.stringify(request),
        contentType: "application/json; charset=utf-8",
        dataType: "text",
        success: null,
    });

    promise.always(function(response) {
    });

    promise.fail(function( jqXHR, textStatus ) {
        console.log("get_project_run_state failed");
        console.log(textStatus);
    });

    promise.done(function (response) {
        console.log("get_project_run_state succeeded");

        try {
            var responseJson = JSON.parse(response);
        } 
        catch (e) {
            console.log("Failed to parse response JSON");
            return;
        };

        if ( (responseJson !== null) && (responseJson['RunComplete'] !== null) ) {
            $('#project-run-progress').prop('hidden', true);
            $('#project-run-complete').prop('hidden', false);
            $('#start-project-run-button').prop("disabled", true);
            $('#stop-project-run-button').prop("disabled", true);
            $('#project-run-reset-button').prop('disabled', false);
        }

        let updates = [];
        if ( responseJson !== null ) {
            updates = responseJson['Updates'];
        }

        updateDataTable('project-run-history-table', updates);
    });

    return promise;
};

function runProject(project) {

    $('#project-run-progress').prop('hidden', false);
    $('#project-run-reset-button').prop('disabled', true);

    let request = { 'ProjectID': project['ID'] }

    var promise = $.ajax({
        url: '/run_project',
        type: 'POST',
        data: JSON.stringify(request),
        contentType: "application/json; charset=utf-8",
        dataType: "text",
        success: null,
    });

    promise.always(function(response) {
    });

    promise.fail(function( jqXHR, textStatus ) {
        console.log("run_project failed");
        $('#project-run-progress').prop('hidden', true);
        $('#start-project-run-button').prop("disabled", false);
        $('#stop-project-run-button').prop("disabled", true);
        $('#project-run-reset-button').prop('disabled', false);
        console.log(textStatus);
    });

    promise.done(function (response) {
        console.log("run_project succeeded");

        try {
            var responseJson = JSON.parse(response);
        } 
        catch (e) {
            console.log("Failed to parse response JSON");
            return;
        };

        let runComplete = responseJson['RunComplete'];

        if ( runComplete === null ) {
            if ( $('#start-project-run-button').prop('disabled') ) {
                // Optimize again
                runProject(project);
            }
            else {
                $('#project-run-progress').prop('hidden', true);
                $('#project-run-reset-button').prop('disabled', false);
            };
        }

        updateProjectRunState(project['ID']);
    });

    return promise;
};

function resetProjectRun(project) {

    $('#modal-spinner').modal('show');

    let request = { 'ProjectID': project['ID'] };

    var promise = $.ajax({
        url: '/reset_project_run',
        type: 'POST',
        data: JSON.stringify(request),
        contentType: "application/json; charset=utf-8",
        dataType: "text",
        success: null,
    });

    promise.always(function(response) {
        $('#modal-spinner').on('shown.bs.modal', function () {
            $('#modal-spinner').modal('hide');
        })
        $('#modal-spinner').modal('hide');
    });

    promise.fail(function( jqXHR, textStatus ) {
        console.log("reset_project_run failed");
        console.log(textStatus);
    });

    promise.done(function (response) {
        console.log("reset_project_run succeeded");

        try {
            var responseJson = JSON.parse(response);
        } 
        catch (e) {
            console.log("Failed to parse response JSON");
            return;
        };

        refreshProjectRun(project['ID']);
    });

    return promise;
};
