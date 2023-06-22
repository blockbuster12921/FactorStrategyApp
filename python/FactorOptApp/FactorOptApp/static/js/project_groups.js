"use strict";

function initProjectGroups() {

    updateProjectGroupsTable();
}

function updateProjectGroupsTable() {

    var promise = $.ajax({
        url: '/get_project_groups',
        type: 'GET',
        data: JSON.stringify({}),
        contentType: "application/json; charset=utf-8",
        dataType: "text",
        success: null,
    });

    promise.always(function(response) {
    });

    promise.fail(function( jqXHR, textStatus ) {
        console.log("get_project_groups failed");
        console.log(textStatus);
    });

    promise.done(function (response) {
        console.log("get_project_groups succeeded");

        try {
            var responseJson = JSON.parse(response);
        } 
        catch (e) {
            console.log("Failed to parse response JSON");
            return;
        };

        initProjectGroupsTable(responseJson);
    });

    return promise;
};

function initProjectGroupsTable(data)
{
    let projectGroupsTable = $('#project-groups-table');
    if ( $.fn.DataTable.isDataTable(projectGroupsTable) ) {
        projectGroupsTable.DataTable().clear().destroy();
        projectGroupsTable.empty();
    }

    function dtRenderActionButtons(data, type, row, meta) {

        let html = "";
        html += '<div class="btn-toolbar d-flex flex-nowrap justify-content-end">';
        html += '<div class="btn-group btn-group-xs pl-1" role="group">';

        html += '<button class="btn btn-primary btn-sm object-table-action-button btn-warning"';
        html += '" data-toggle="modal" data-target="#delete-project-group-modal"';
        html += '" data-project-group-id="' + row['ID'] + '" data-project-group-name="' + row['Name'] + '"><i class="'
        html += 'far fa-trash-alt';
        html += '" aria-hidden="true"></i> ';
        html += 'Delete';
        html += '</button>';

        html += '</div>';
        html += '</div>';
        return html;
    }

    let columns = [
        { data: "Name", title: "Name", className: "all", render: function(data, type, row, meta) { return dtRenderHyperlink(data, type, row, meta, row['URL'], false) } },
    ];
    columns.push({ data: null, searchable: false, orderable: false, className: "all", render: dtRenderActionButtons });

    initDataTable('project-groups-table', data, columns, "project groups", [[1, "asc"]], '');
}


function registerDeleteProjectGroupEvents() {
    $('#delete-project-group-modal').on('show.bs.modal', function (event) {
        let button = $(event.relatedTarget);
        const projectGroupId = button.data('projectGroupId');
        const projectGroupName = button.data('projectGroupName');

        $('#delete-project-group-modal-submit-button').off();
        $('#delete-project-group-modal-submit-button').on('click', function(){
            deleteProjectGroup(projectGroupId);
        });

        $('#delete-project-group-modal-project-name').text(projectGroupName);
    })
}

function deleteProjectGroup(projectGroupId) {

    let request = { 'ProjectGroupID': projectGroupId };

    var promise = $.ajax({
        url: '/delete_project_group',
        type: 'POST',
        data: JSON.stringify(request),
        contentType: "application/json; charset=utf-8",
        dataType: "text",
        success: null,
    });

    promise.fail(function( jqXHR, textStatus ) {
        console.log("delete_project_group failed");
        console.log(textStatus);
        return;
    });

    promise.done(function (response) {
        console.log("delete_project_group succeeded");
        updateProjectGroupsTable();
    });
}
