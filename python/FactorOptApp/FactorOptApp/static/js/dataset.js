"use strict";

function initDataset(dataset, referenceData, docTitlePrefix) {

    const datasetId = dataset['ID'];

    document.title = docTitlePrefix + dataset['Name'];

    $('#dataset-name-input').on('change', function () {
        updateDatasetMetadata(datasetId);
        $("li.breadcrumb-item.active").text($('#dataset-name-input').val());
        document.title = docTitlePrefix + $('#dataset-name-input').val();
    });

    $('#dataset-region-select').on('change', function () {
        updateDatasetSectorSelect(referenceData);
        updateDatasetMetadata(datasetId);
    });

    updateDatasetSectorSelect(referenceData, dataset['SectorID']);
    $('#dataset-sector-select').on('change', function () {
        updateDatasetMetadata(datasetId, null);
    });

    $('#dataset-notes-input').on('change', function () {
        updateDatasetMetadata(datasetId);
    });

    updateDatasetDataInfo(datasetId);

    $('#load-dataset-data-file-input').on('change', function() { 
      loadDatasetData(datasetId);
    });
}

function updateDatasetSectorSelect(referenceData, sectorId) {

    let html = '';

    const regionId = $('#dataset-region-select').val();
    referenceData['Regions'].forEach(function(region){
        if ( region['ID'] !== regionId ) {
            return;
        }
        region['Sectors'].forEach(function(sector){
            html += '<option value="' + sector['ID'] + '">' + sector['Name'] + '</option>';
        });
    });

    $('#dataset-sector-select').html(html).selectpicker('refresh');

    if ( sectorId !== null ) {
        $('#dataset-sector-select').selectpicker('val', sectorId);
    }
}

function updateDatasetMetadata(datasetId) {

    let request = {
        'ID': datasetId,
        'Name': $('#dataset-name-input').val(),
        'Notes': $('#dataset-notes-input').val(),
        'RegionID': $('#dataset-region-select').val(),
        'SectorID': $('#dataset-sector-select').val(),
    };
    console.log(request);

    var promise = $.ajax({
        url: '/update_dataset_metadata',
        type: 'PUT',
        data: JSON.stringify(request),
        contentType: "application/json; charset=utf-8",
        dataType: "text",
        success: null,
    });

    promise.fail(function( jqXHR, textStatus ) {
        console.log("update_dataset_metadata failed");
        console.log(textStatus);
        return;
    });

    promise.done(function (response) {
        console.log("update_dataset_metadata succeeded");
        console.log(response);
    });

    promise.always(function (reponse) {
    });
}

// Load dataset data from Excel file
function loadDatasetData(datasetId) {

    $('#load-dataset-data-alerts-container').empty();
    $('#load-dataset-data-progress').removeAttr('hidden');
    $('#dataset-data-info').prop('hidden', true);
    $('#dataset-data-info-detail').empty();

    let formData = new FormData();
    formData.append('file', $('#load-dataset-data-file-input').prop('files')[0]);
    formData.append('datasetId', datasetId);

    var promise = $.ajax({
        type: 'POST',
        url:  '/load_dataset_data',
        data: formData,
        contentType: false,
        cache: false,
        processData: false,
        success: null,
    });

    promise.fail(function (response) {
        console.log("load_dataset_data failed");

        var alert = "";
        if ( response.responseText.length === 0 ) {
            alert = "Failed to load data";
        }
        else {
            alert = "Error: " + $('<div/>').text(response.responseText).html();
        };
        $(renderAlert(alert, "danger")).appendTo('#load-dataset-data-alerts-container');
    });

    promise.done(function (response) {
        console.log("load_dataset_data succeeded");
        var alert = "Loaded " + response['FactorCount'].toString() + " factors for ";
        alert += response['StockCount'].toString() + " stocks";
        $(renderAlert(alert, "success")).appendTo('#load-dataset-data-alerts-container');
    });

    promise.always(function() {
        $('#load-dataset-data-progress').prop('hidden', true);
        $('#load-dataset-data-file-input').val(null); // Reset attribute to allow same file to be re-loaded
        updateDatasetDataInfo(datasetId);
    });

    return promise;
};


function getDatasetDataInfo(datasetId) {

    var promise = $.ajax({
        url: '/get_dataset_data_info',
        type: 'POST',
        data: JSON.stringify({'DatasetID': datasetId}),
        contentType: "application/json; charset=utf-8",
        dataType: "text",
        success: null,
    });

    promise.fail(function( jqXHR, textStatus ) {
        console.log("get_dataset_data_info failed");
        console.log(textStatus);
    });

    promise.done(function (response) {
        console.log("get_dataset_data_info succeeded");
    });

    return promise;
}

function updateDatasetDataInfo(datasetId) {

    var promise = getDatasetDataInfo(datasetId);

    promise.fail(function() {
        $('#dataset-data-info-detail').empty();
        var html = '<span class="text-danger">Failed to get data summary</span>';
        $(html).appendTo('#dataset-data-info-detail');
        $('#dataset-data-info').removeAttr('hidden');
        return;
    });

    promise.done(function (response) {

        try {
            var responseJson = JSON.parse(response);
        } 
        catch (e) {
            console.log("Failed to parse response JSON");
            return;
        };

        let html = "";
        if ( responseJson !== null ) {
            html += '<div>Data loaded from "' + responseJson['Filename'] + '"';
            html += ' on ' + moment.utc(responseJson['Created']).local().format("DD MMM YYYY HH:mm:ss") + '</div>';
            html += '<div>' + responseJson['Factors'].length.toString() + ' Factors</div>';
            html += '<div>' + responseJson['Stocks'].length.toString() + ' Stocks</div>';
            html += '<div>Date Range: ' + moment(responseJson['Dates'][0]).format("MMM Y");
            html += ' to ' + moment(responseJson['Dates'][responseJson['Dates'].length-1]).format("MMM Y") + '</div>';
        }
        else {
            html += '<div>No data loaded</div>';
        }

        $(html).appendTo('#dataset-data-info-detail');
        $('#dataset-data-info').removeAttr('hidden');
    });
}
