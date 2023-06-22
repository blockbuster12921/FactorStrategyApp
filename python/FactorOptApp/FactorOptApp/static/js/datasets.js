"use strict";

function initDatasets() {

    updateDatasetsTable();

    $('#create-dataset-button').on('click', function(){
        createDataset();
    });
}

function updateDatasetsTable() {

    var promise = $.ajax({
        url: '/get_datasets',
        type: 'GET',
        data: JSON.stringify({}),
        contentType: "application/json; charset=utf-8",
        dataType: "text",
        success: null,
    });

    promise.always(function(response) {
    });

    promise.fail(function( jqXHR, textStatus ) {
        console.log("get_datasets failed");
        console.log(textStatus);
    });

    promise.done(function (response) {
        console.log("get_datasets succeeded");

        try {
            var responseJson = JSON.parse(response);
        } 
        catch (e) {
            console.log("Failed to parse response JSON");
            return;
        };

        initDatasetsTable(responseJson);
    });

    return promise;
};

function initDatasetsTable(data)
{
    let datasetsTable = $('#datasets-table');
    if ( $.fn.DataTable.isDataTable(datasetsTable) ) {
        datasetsTable.DataTable().clear().destroy();
        datasetsTable.empty();
    }

    function dtRenderActionButtons(data, type, row, meta) {

        let html = "";
        html += '<div class="btn-toolbar d-flex flex-nowrap justify-content-end">';
        html += '<div class="btn-group btn-group-xs pl-1" role="group">';

        html += '<button class="btn btn-primary btn-sm object-table-action-button btn-warning"';
        html += '" data-toggle="modal" data-target="#delete-dataset-modal"';
        html += '" data-dataset-id="' + row['ID'] + '" data-dataset-name="' + row['Name'] + '"><i class="'
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
        { data: "Region", title: "Region", className: "all", },  
        { data: "Sector", title: "Sector", className: "all", },
        { data: "StockCount", title: "Stocks", className: "all", },
        { data: "FactorCount", title: "Factors", className: "all", },
        { data: "DataStartDate", title: "First Data Month", className: "all", render: dtRenderMonth },
        { data: "DataEndDate", title: "Last Data Month", className: "all", render: dtRenderMonth },
        { data: "DataLoadedTimestamp", title: "Data Loaded", className: "all", render: dtRenderTimestamp },
    ];
    columns.push({ data: null, searchable: false, orderable: false, className: "all", render: dtRenderActionButtons });

    initDataTable('datasets-table', data, columns, "datasets", [[1, "asc"]], '');
}


function registerDeleteDatasetEvents() {
    $('#delete-dataset-modal').on('show.bs.modal', function (event) {
        let button = $(event.relatedTarget);
        const datasetId = button.data('datasetId');
        const datasetName = button.data('datasetName');

        $('#delete-dataset-modal-submit-button').off();
        $('#delete-dataset-modal-submit-button').on('click', function(){
            deleteDataset(datasetId);
        });

        $('#delete-dataset-modal-dataset-name').text(datasetName);
    })
}

function deleteDataset(datasetId) {

    let request = { 'DatasetID': datasetId };

    var promise = $.ajax({
        url: '/delete_dataset',
        type: 'POST',
        data: JSON.stringify(request),
        contentType: "application/json; charset=utf-8",
        dataType: "text",
        success: null,
    });

    promise.fail(function( jqXHR, textStatus ) {
        console.log("delete_dataset failed");
        console.log(textStatus);
        return;
    });

    promise.done(function (response) {
        console.log("delete_dataset succeeded");
        updateDatasetsTable();
    });
}

function createDataset() {

    var promise = $.ajax({
        url: '/create_dataset',
        type: 'POST',
        data: JSON.stringify({}),
        contentType: "application/json; charset=utf-8",
        dataType: "text",
        success: null,
    });

    promise.fail(function( jqXHR, textStatus ) {
        console.log("create_dataset failed");
        console.log(textStatus);
        $('#modal-spinner').modal('hide');
        return;
    });

    promise.done(function (response) {
        console.log("create_dataset succeeded");

        try {
            var responseJson = JSON.parse(response);
        } 
        catch (e) {
            console.log("Failed to parse response JSON");
            return;
        };

        window.location.href = window.location.origin + responseJson['DatasetURL'];
    });

}
