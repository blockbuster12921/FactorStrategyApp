"use strict";

function initProjects(projects) {

    $('#show-archived-projects-check-input').on('change', function(){
        updateProjectsTable();
    });

    updateProjectsTable();

    $('#create-project-group-button').on('click', function(){
        createProjectGroupFromSelectedProjects();
    });
}

function updateProjectsTable() {

    var promise = $.ajax({
        url: '/get_projects',
        type: 'POST',
        data: JSON.stringify({}),
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

        initProjectsTable(responseJson);
    });

    return promise;
};

function initProjectsTable(data)
{
    let projectsTable = $('#projects-table');
    if ( $.fn.DataTable.isDataTable(projectsTable) ) {
        projectsTable.DataTable().clear().destroy();
        projectsTable.empty();
    }

    const showArchived = $('#show-archived-projects-check-input').is(':checked');

    function dtRenderActionButtons(data, type, row, meta) {
        const archived = ('Deleted' in row);

        let html = "";
        html += '<div class="btn-toolbar d-flex flex-nowrap justify-content-end">';
        html += '<div class="btn-group btn-group-xs pl-1" role="group">';

        html += '<button class="btn btn-primary btn-sm object-table-action-button clone-project-button mr-2"';
        html += '" data-project-id="' + row['ID'] + '"><i class="fas fa-clone" aria-hidden="true"></i> ';
        html += 'Clone';
        html += '</button>';

        if ( archived ) {
            html += '<button class="btn btn-primary btn-sm object-table-action-button restore-project-button mr-2"';
            html += '" data-project-id="' + row['ID'] + '"><i class="fa fa-undo" aria-hidden="true"></i> ';
            html += 'Restore';
            html += '</button>';
        }

        html += '<button class="btn btn-primary btn-sm object-table-action-button';
        if ( archived ) {
            html += ' btn-warning';
        }
        else {
            html += ' archive-project-button';
        }
        html += '"';
        if ( archived ) {
            html += '" data-toggle="modal" data-target="#delete-project-modal"';
        }
        html += '" data-project-id="' + row['ID'] + '" data-project-name="' + row['Name'] + '"><i class="'
        html += archived ? 'far fa-trash-alt' : 'fa fa-archive';
        html += '" aria-hidden="true"></i> ';
        html += archived ? 'Delete' : 'Archive';
        html += '</button>';

        html += '</div>';
        html += '</div>';
        return html;
    }

    function dtRenderArchivedStatus(data, type, row, meta) {
        if ( (data === undefined) || (data === null) ) {
            return "";
        }
        return '<i class="fa fa-check" aria-hidden="true"></i>';
    }

    function dtRenderSelectedCheckbox(data, type, row, meta) {
        if (type != 'display' ) {
            return data;
        }

        let html = "";
        html += '<button class="btn btn-light table-checkbox-button projects-table-select-checkbox">';
        html += '<i class="far fa' + (data ? '-check' : '') + '-square" aria-hidden="true"></i></button>';
        return html;
    }

    let columns = [
        { data: "selected", defaultContent: false, title: "", searchable: false, orderable: false, className: "all", render: dtRenderSelectedCheckbox, className: "all" }, // Action buttons
        { data: "Name", title: "Name", className: "all", render: function(data, type, row, meta) { return dtRenderHyperlink(data, type, row, meta, row['URL'], false) } },
        { data: "Status", title: "Status", className: "all", }, 
        { data: "DatasetNames", title: "Datasets", classname: "all" },
    ];
    if ( showArchived ) {
        columns.push({ data: "Deleted", title: "Archived?", className: "all", render: dtRenderArchivedStatus });
    }
    columns.push({ data: null, searchable: false, orderable: false, className: "all", render: dtRenderActionButtons });

    let filteredData = data;
    if ( ! showArchived ) {
        filteredData = [];
        data.forEach(row => {
            if ( (row['Deleted'] === undefined) || (row['Deleted'] === null) ) {
                filteredData.push(row);
            }
        });
    }

    initDataTable('projects-table', filteredData, columns, "projects", [[1, "asc"]], '');

    $('.clone-project-button').on('click', function(){
        const projectId = $(this).data('projectId');
        cloneProject(projectId);
    });

    $('.archive-project-button').on('click', function(){
        const projectId = $(this).data('projectId');
        archiveProject(projectId);
    });

    $('.restore-project-button').on('click', function(){
        const projectId = $(this).data('projectId');
        restoreProject(projectId);
    });

    $('#projects-table tbody').on('click', '.projects-table-select-checkbox', function(){
        let button = $(this);
        let cell = $('#projects-table').DataTable().cell(button.parent());
        let enabled = cell.data();
        cell.data(!enabled);
        button.find('i').toggleClass('fa-check-square fa-square');
    });

}

function createProjectGroupFromSelectedProjects() {

    let selectedProjectIDs = [];
    $('#projects-table').DataTable().rows().every(function(index, element) {
        const data = this.data();

        const selected = data['selected'];
        if ( selected ) {
            selectedProjectIDs.push(data['ID']);
        }
    });

    if ( selectedProjectIDs.length < 2 ) {
        return;
    }

    let request = { 'ProjectIDs': selectedProjectIDs };

    var promise = $.ajax({
        url: '/create_project_group',
        type: 'POST',
        data: JSON.stringify(request),
        contentType: "application/json; charset=utf-8",
        dataType: "text",
        success: null,
    });

    promise.fail(function( jqXHR, textStatus ) {
        console.log("create_project_group failed");
        console.log(textStatus);
        $('#modal-spinner').modal('hide');
        return;
    });

    promise.done(function (response) {
        console.log("create_project_group succeeded");

        try {
            var responseJson = JSON.parse(response);
        } 
        catch (e) {
            console.log("Failed to parse response JSON");
            return;
        };

        window.location.href = window.location.origin + responseJson['ProjectGroupURL'];
    });

}

function registerDeleteProjectEvents() {
    $('#delete-project-modal').on('show.bs.modal', function (event) {
        let button = $(event.relatedTarget);
        const projectId = button.data('projectId');
        const projectName = button.data('projectName');
        console.log(projectId);

        $('#delete-project-modal-submit-button').off();
        $('#delete-project-modal-submit-button').on('click', function(){
            deleteProject(projectId);
        });

        $('#delete-project-modal-project-name').text(projectName);
    })
}

function cloneProject(projectId) {

    let request = { 'ProjectID': projectId };

    var promise = $.ajax({
        url: '/clone_project',
        type: 'POST',
        data: JSON.stringify(request),
        contentType: "application/json; charset=utf-8",
        dataType: "text",
        success: null,
    });

    promise.fail(function( jqXHR, textStatus ) {
        console.log("clone_project failed");
        console.log(textStatus);
    });

    promise.done(function (response) {
        console.log("clone_project succeeded");

        try {
            var responseJson = JSON.parse(response);
        } 
        catch (e) {
            console.log("Failed to parse response JSON");
            updateProjectsTable();
            return;
        };

        window.location.href = window.location.origin + responseJson['ProjectURL'];
    });

    return promise;
}

function deleteProject(projectId) {

    let request = { 'ProjectID': projectId };

    var promise = $.ajax({
        url: '/delete_project',
        type: 'POST',
        data: JSON.stringify(request),
        contentType: "application/json; charset=utf-8",
        dataType: "text",
        success: null,
    });

    promise.fail(function( jqXHR, textStatus ) {
        console.log("delete_project failed");
        console.log(textStatus);
    });

    promise.done(function (response) {
        console.log("delete_project succeeded");
        updateProjectsTable();
    });

    return promise;
}

function archiveProject(projectId) {

    let request = { 'ProjectID': projectId };

    var promise = $.ajax({
        url: '/archive_project',
        type: 'POST',
        data: JSON.stringify(request),
        contentType: "application/json; charset=utf-8",
        dataType: "text",
        success: null,
    });

    promise.fail(function( jqXHR, textStatus ) {
        console.log("archive_project failed");
        console.log(textStatus);
    });

    promise.done(function (response) {
        console.log("archive_project succeeded");
        updateProjectsTable();
    });

    return promise;
}

function restoreProject(projectId) {

    let request = { 'ProjectID': projectId };

    var promise = $.ajax({
        url: '/restore_project',
        type: 'POST',
        data: JSON.stringify(request),
        contentType: "application/json; charset=utf-8",
        dataType: "text",
        success: null,
    });

    promise.fail(function( jqXHR, textStatus ) {
        console.log("restore_project failed");
        console.log(textStatus);
        return;
    });

    promise.done(function (response) {
        console.log("restore_project succeeded");
        updateProjectsTable();
    });
}
