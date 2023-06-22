"use strict";

function initProjectStocks(project) {

    const projectId = project['ID'];

    initStocksTable(projectId);

    refreshProjectStocks(projectId);

    $('#project-reset-stocks-enabled-button').on('click', function(){
        resetStocksEnabledStatus(projectId);
        resetStocksForwardMonthEnabledStatus(projectId);
    });
}

function refreshProjectStocks(projectId) {

    var promise = $.ajax({
        url: '/get_project_stocks_info',
        type: 'POST',
        data: JSON.stringify({'ProjectID': projectId}),
        contentType: "application/json; charset=utf-8",
        dataType: "text",
        success: null,
    });

    promise.fail(function( jqXHR, textStatus ) {
        console.log("get_project_stocks_info failed");
        console.log(textStatus);
        return;
    });

    promise.done(function (response) {
        console.log("get_project_stocks_info succeeded");

        try {
            var responseJson = JSON.parse(response);
        } 
        catch (e) {
            console.log("Failed to parse response JSON");
            return;
        };

        if ( responseJson !== null ) {
            updateDataTable('stocks-table', responseJson['Stocks']);
        }
        else {
            updateDataTable('stocks-table', []);
        }
    });

    return promise;
}

function initStocksTable(projectId) {

    function dtRenderStockCheckbox(data, type, row, meta, cls) {
        if (type != 'display' ) {
            return data;
        }

        let html = "";
        html += '<button class="btn btn-light table-checkbox-button ' + cls + '">';
        html += '<i class="far fa' + (data ? '-check' : '') + '-square" aria-hidden="true"></i></button>';
        return html;
    }

    function dtRenderStockEnabledCheckbox(data, type, row, meta) {
        return dtRenderStockCheckbox(data, type, row, meta, "project-stock-enabled-checkbox");
    }

    function dtRenderStockForwardMonthEnabledCheckbox(data, type, row, meta) {
        return dtRenderStockCheckbox(data, type, row, meta, "project-stock-forward-month-enabled-checkbox");
    }

    const columns = [
        { data: "Enabled", title: 'Enabled?', searchable: false, orderable: true, className: "all", render: dtRenderStockEnabledCheckbox, className: "all" }, // Action buttons
        { data: "ForwardMonthEnabled", title: 'Enabled for Forward Month?', searchable: false, orderable: true, className: "all", render: dtRenderStockForwardMonthEnabledCheckbox, className: "all" }, // Action buttons
        { data: "Name", title: "Name", className: "all", },
        { data: "Ticker", title: "Ticker", className: "all", },  
        { data: "DatasetName", title: "Dataset", className: "all", },
        { data: "SubSector", title: "Sub-Sector", className: "all", },  
    ]
    initDataTable('stocks-table', [], columns, "stocks", [[0, "asc"]], '');

    $('#stocks-table tbody').on('click', '.project-stock-enabled-checkbox', function(){
        let button = $(this);
        let cell = $('#stocks-table').DataTable().cell(button.parent());
        let enabled = cell.data();
        cell.data(!enabled);
        updateStocksEnabledStatus(projectId).done(function(){
            button.find('i').toggleClass('fa-check-square fa-square');
        });
    });

    $('#stocks-table tbody').on('click', '.project-stock-forward-month-enabled-checkbox', function(){
        let button = $(this);
        let cell = $('#stocks-table').DataTable().cell(button.parent());
        let enabled = cell.data();
        cell.data(!enabled);
        updateStocksForwardMonthEnabledStatus(projectId).done(function(){
            button.find('i').toggleClass('fa-check-square fa-square');
        });
    });

}

function updateStocksEnabledStatus(projectId) {

    let disabled = [];
    $('#stocks-table').DataTable().rows().every( function ( rowIdx, tableLoop, rowLoop ) {
        const data = this.data();
        if ( ! data.Enabled ) {
            disabled.push({'DatasetID': data.DatasetID, 'DatasetIndex': data.DatasetIndex})
        }
    } );

    let request = { 'ProjectID': projectId, 'StocksDisabled': disabled };

    var promise = $.ajax({
        url: '/set_project_stocks_disabled_status',
        type: 'POST',
        data: JSON.stringify(request),
        contentType: "application/json; charset=utf-8",
        dataType: "text",
        success: null,
    });

    promise.fail(function( jqXHR, textStatus ) {
        console.log("set_project_stocks_disabled_status failed");
        console.log(textStatus);
    });

    promise.done(function (response) {
        console.log("set_project_stocks_disabled_status succeeded");
    });

    return promise;
};

function resetStocksEnabledStatus(projectId) {

    let request = { 'ProjectID': projectId, 'StocksDisabled': [] };

    var promise = $.ajax({
        url: '/set_project_stocks_disabled_status',
        type: 'POST',
        data: JSON.stringify(request),
        contentType: "application/json; charset=utf-8",
        dataType: "text",
        success: null,
    });

    promise.fail(function( jqXHR, textStatus ) {
        console.log("set_project_stocks_disabled_status failed");
        console.log(textStatus);
    });

    promise.done(function (response) {
        console.log("set_project_stocks_disabled_status succeeded");
        refreshProjectStocks(projectId);
    });

    return promise;
};

function updateStocksForwardMonthEnabledStatus(projectId) {

    let disabled = [];
    $('#stocks-table').DataTable().rows().every( function ( rowIdx, tableLoop, rowLoop ) {
        const data = this.data();
        if ( ! data.ForwardMonthEnabled ) {
            disabled.push({'DatasetID': data.DatasetID, 'DatasetIndex': data.DatasetIndex})
        }
    } );

    let request = { 'ProjectID': projectId, 'StocksDisabled': disabled };

    var promise = $.ajax({
        url: '/set_project_stocks_forward_month_disabled_status',
        type: 'POST',
        data: JSON.stringify(request),
        contentType: "application/json; charset=utf-8",
        dataType: "text",
        success: null,
    });

    promise.fail(function( jqXHR, textStatus ) {
        console.log("set_project_stocks_forward_month_disabled_status failed");
        console.log(textStatus);
    });

    promise.done(function (response) {
        console.log("set_project_stocks_forward_month_disabled_status succeeded");
    });

    return promise;
};

function resetStocksForwardMonthEnabledStatus(projectId) {

    let request = { 'ProjectID': projectId, 'StocksDisabled': [] };

    var promise = $.ajax({
        url: '/set_project_stocks_forward_month_disabled_status',
        type: 'POST',
        data: JSON.stringify(request),
        contentType: "application/json; charset=utf-8",
        dataType: "text",
        success: null,
    });

    promise.fail(function( jqXHR, textStatus ) {
        console.log("set_project_stocks_forward_month_disabled_status failed");
        console.log(textStatus);
    });

    promise.done(function (response) {
        console.log("set_project_stocks_forward_month_disabled_status succeeded");
        refreshProjectStocks(projectId);
    });

    return promise;
};
