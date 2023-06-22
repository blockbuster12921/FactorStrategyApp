"use strict";

function initProjectGroupRun(projectGroup, factorStrategies) {

    $('#start-project-group-run-button').on('click', function() { 
        $('#start-project-group-run-button').prop("disabled", true);
        $('#stop-project-group-run-button').prop("disabled", false);
        runProjectGroup(projectGroup);
    });

    $('#stop-project-group-run-button').on('click', function() { 
        $('#start-project-group-run-button').prop("disabled", false);
        $('#stop-project-group-run-button').prop("disabled", true);
    });

    $('#project-group-run-reset-button').on('click', function() { 
        resetProjectGroupRun(projectGroup);
    });

    $('#project-group-excel-export-stock-ranking-button').on('click', function() { 
        exportProjectGroupStockRanking(projectGroup, $('#project-group-excel-export-stock-ranking-button').find(".button-progress-spinner"));
    });

    let html = '';
    factorStrategies.forEach(function(strategy, index) {
        html += '<option value="' + strategy['ID'] + '">';
        html += 'Combinations: ' + strategy['Description']['combinations'] + '; ';
        html += 'Factors: ' + strategy['Description']['factors'] + '; ';
        html += strategy['Description']['weighting'] + ' Weighting';
        html += '</option>';
    });
    $('#project-group-stock-ranking-report-strategy-select').html(html).selectpicker('refresh');
    if ( factorStrategies.length > 0 ) {
        $('#project-group-stock-ranking-report-strategy-select').selectpicker('val', '8fe1a3e1-fb3f-416e-b5d3-6fae59587ee1');
    }

    initProjectGroupRunHistoryTable();

    updateProjectGroupRunState(projectGroup);
}

function refreshProjectGroupRun(projectGroup) {
    $('#project-group-run-progress').prop('hidden', true);
    $('#project-group-run-complete').prop('hidden', true);
    $('#start-project-group-run-button').prop("disabled", false);
    $('#stop-project-group-run-button').prop("disabled", true);
    updateProjectGroupRunState(projectGroup);
}

function initProjectGroupRunHistoryTable() {

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
        { data: "ProjectName", title: "Project", className: "all" },
        { data: "Timestamp", title: "Time", className: "all", render: dtRenderTimestamp },
        { data: "Stage", title: "Stage", className: "all", render: dtRenderStage },  
        { data: "Status", title: "Status", className: "all", render: dtRenderStatus },  
        { data: "TargetMonth", title: "Target Month", className: "all", render: dtRenderMonth },
        { data: "Detail", title: "Detail", className: "all", defaultContent: "" },
    ]
    initDataTable('project-group-run-history-table', [], columns, "items", [[1, "desc"]], '');
}

function updateProjectGroupRunState(projectGroup) {

    let request = { 'ProjectGroupID': projectGroup['ID'] };

    var promise = $.ajax({
        url: '/get_project_group_run_state',
        type: 'PUT',
        data: JSON.stringify(request),
        contentType: "application/json; charset=utf-8",
        dataType: "text",
        success: null,
    });

    promise.always(function(response) {
    });

    promise.fail(function( jqXHR, textStatus ) {
        console.log("get_project_group_run_state failed");
        console.log(textStatus);
    });

    promise.done(function (response) {
        console.log("get_project_group_run_state succeeded");

        try {
            var responseJson = JSON.parse(response);
        } 
        catch (e) {
            console.log("Failed to parse response JSON");
            return;
        };

        let updates = [];
        if ( responseJson !== null ) {
            let projectCount = responseJson.length;
            let completeCount = 0;

            responseJson.forEach(function(project) {

                if ( project['ProjectRunState'] === null ) {
                    return;
                }

                if ( (project['ProjectRunState']['RunComplete'] !== null) && (project['ProjectRunState']['RunComplete'] !== undefined) ) {
                    completeCount++;
                }

                project['ProjectRunState']['Updates'].forEach(function(update){
                    update['ProjectName'] = project['ProjectName'];
                    updates.push(update);
                })
            });

            if ( completeCount === projectCount ) {
                $('#project-group-run-progress').prop('hidden', true);
                $('#project-group-run-complete').prop('hidden', false);
                $('#start-project-group-run-button').prop("disabled", true);
                $('#stop-project-group-run-button').prop("disabled", true);
                $('#project-group-run-reset-button').prop('disabled', false);
            }
        }

        updateDataTable('project-group-run-history-table', updates);
    });

    return promise;
};

function runProjectGroup(projectGroup) {

    $('#project-group-run-progress').prop('hidden', false);
    $('#project-group-run-reset-button').prop('disabled', true);

    let request = { 'ProjectGroupID': projectGroup['ID'] }

    var promise = $.ajax({
        url: '/run_project_group',
        type: 'POST',
        data: JSON.stringify(request),
        contentType: "application/json; charset=utf-8",
        dataType: "text",
        success: null,
    });

    promise.always(function(response) {
    });

    promise.fail(function( jqXHR, textStatus ) {
        console.log("run_project_group failed");
        $('#project-group-run-progress').prop('hidden', true);
        $('#start-project-group-run-button').prop("disabled", false);
        $('#stop-project-group-run-button').prop("disabled", true);
        $('#project-group-run-reset-button').prop('disabled', false);
        console.log(textStatus);
    });

    promise.done(function (response) {
        console.log("run_project_group succeeded");

        try {
            var responseJson = JSON.parse(response);
        } 
        catch (e) {
            console.log("Failed to parse response JSON");
            return;
        };

        let runComplete = responseJson['RunComplete'];

        if ( ! runComplete ) {
            if ( $('#start-project-group-run-button').prop('disabled') ) {
                // Optimize again
                runProjectGroup(projectGroup);
            }
            else {
                $('#project-group-run-progress').prop('hidden', true);
                $('#project-group-run-reset-button').prop('disabled', false);
            };
        }

        updateProjectGroupRunState(projectGroup);
    });

    return promise;
};

function resetProjectGroupRun(projectGroup) {

    $('#modal-spinner').modal('show');

    let request = { 'ProjectGroupID': projectGroup['ID'] };

    var promise = $.ajax({
        url: '/reset_project_group_run',
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
        console.log("reset_project_group_run failed");
        console.log(textStatus);
        $('#modal-spinner').modal('hide');
    });

    promise.done(function (response) {
        console.log("reset_project_group_run succeeded");

        refreshProjectGroupRun(projectGroup);
    });

    return promise;
};
