"use strict";

function initProjectGroup(projectGroup, docTitlePrefix) {

    const projectGroupId = projectGroup['ID'];

    document.title = docTitlePrefix + projectGroup['Name'];

    $('#project-group-name-input').on('change', function () {
        updateProjectGroupMetadata(projectGroupId);
        $("li.breadcrumb-item.active").text($('#project-group-name-input').val());
        document.title = docTitlePrefix + $('#project-group-name-input').val();
    });

    $('#project-group-notes-input').on('change', function () {
        updateProjectGroupMetadata(projectGroupId);
    });

    updateProjectGroupProjectsTable(projectGroup);
}

function updateProjectGroupMetadata(projectGroupId) {

    let request = {
        'ID': projectGroupId,
        'Name': $('#project-group-name-input').val(),
        'Notes': $('#project-group-notes-input').val(),
    };
    console.log(request);

    var promise = $.ajax({
        url: '/update_project_group_metadata',
        type: 'PUT',
        data: JSON.stringify(request),
        contentType: "application/json; charset=utf-8",
        dataType: "text",
        success: null,
    });

    promise.fail(function( jqXHR, textStatus ) {
        console.log("update_project_group_metadata failed");
        console.log(textStatus);
        return;
    });

    promise.done(function (response) {
        console.log("update_project_group_metadata succeeded");
        console.log(response);
    });

    promise.always(function (reponse) {
    });
}


function updateProjectGroupProjectsTable(projectGroup) {

    const request = { 
        'ProjectIDs': projectGroup['ProjectIDs'],
        'DataInfo': true,
    };

    var promise = $.ajax({
        url: '/get_projects',
        type: 'POST',
        data: JSON.stringify(request),
        contentType: "application/json; charset=utf-8",
        dataType: "text",
        success: null,
    });

    promise.always(function(response) {
    });

    promise.fail(function( jqXHR, textStatus ) {
        console.log("get_projects failed");
        console.log(textStatus);
    });

    promise.done(function (response) {
        console.log("get_projects succeeded");

        try {
            var responseJson = JSON.parse(response);
        } 
        catch (e) {
            console.log("Failed to parse response JSON");
            return;
        };

        initProjectGroupProjectsTable(responseJson);
    });

    return promise;
};

function initProjectGroupProjectsTable(data)
{
    let projectsTable = $('#project-group-projects-table');
    if ( $.fn.DataTable.isDataTable(projectsTable) ) {
        projectsTable.DataTable().clear().destroy();
        projectsTable.empty();
    }

    let columns = [
        { data: "Name", title: "Name", className: "all", render: function(data, type, row, meta) { return dtRenderHyperlink(data, type, row, meta, row['URL'], true) } },
        { data: "Status", title: "Status", className: "all", },  
        { data: "DatasetNames", title: "Datasets", classname: "all" },
        { data: "StockCount", title: "Stocks", className: "all", },
        { data: "DataEndDate", title: "Last Data Month", className: "all", render: dtRenderMonth },
    ];

    initDataTable('project-group-projects-table', data, columns, "projects", [[0, "asc"]], null);
}

