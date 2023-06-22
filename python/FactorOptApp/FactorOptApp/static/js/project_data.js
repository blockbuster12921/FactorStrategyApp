"use strict";

function initData(project, datasets, docTitlePrefix) {

    const projectId = project['ID'];

    document.title = docTitlePrefix + project['Name'];

    $('#project-name-input').on('change', function () {
        updateProjectMetadata(projectId);
        $("li.breadcrumb-item.active").text($('#project-name-input').val());
        document.title = docTitlePrefix + $('#project-name-input').val();
    });

    $('#project-status-select').on('change', function () {
        updateProjectMetadata(projectId);
    });

    updateProjectDatasetsSelect(project, datasets);
    $('#project-datasets-select').on('change', function () {
        updateProjectMetadata(projectId, null).always(function() {
            refreshProjectStocks(projectId);
            refreshProjectFactorsInfo(projectId);
        });
    });

    $('#project-notes-input').on('change', function () {
        updateProjectMetadata(projectId);
    });
}

function updateProjectDatasetsSelect(project, datasets) {

    let html = '';
    datasets.forEach(function(dataset){
        html += '<option value="' + dataset['ID'] + '">' + dataset['Name'] + '</option>';
    });

    $('#project-datasets-select').html(html).selectpicker('refresh');

    $('#project-datasets-select').selectpicker('val', project['DatasetIDs']);
}

function updateProjectMetadata(projectId) {

    let request = {
        'ID': projectId,
        'Name': $('#project-name-input').val(),
        'Notes': $('#project-notes-input').val(),
        'Status': $('#project-status-select').val(),
        'OOSStartDate': moment().set({
                'year': parseInt($('#project-settings-oos-start-year-select').val()), 
                'month': parseInt($('#project-settings-oos-start-month-select').val())-1,
            }).endOf('month').format("Y-MM-DD"),
        'OOSEndDate': moment().set({
                'year': parseInt($('#project-settings-oos-end-year-select').val()), 
                'month': parseInt($('#project-settings-oos-end-month-select').val())-1,
            }).endOf('month').format("Y-MM-DD"),
        'DatasetIDs': $('#project-datasets-select').val(),
    };
    console.log(request);

    var promise = $.ajax({
        url: '/update_project_metadata',
        type: 'PUT',
        data: JSON.stringify(request),
        contentType: "application/json; charset=utf-8",
        dataType: "text",
        success: null,
    });

    promise.fail(function( jqXHR, textStatus ) {
        console.log("update_project_metadata failed");
        console.log(textStatus);
    });

    promise.done(function (response) {
        console.log("update_project_metadata succeeded");
        console.log(response);
    });

    promise.always(function (reponse) {
    });

    return promise;
}
