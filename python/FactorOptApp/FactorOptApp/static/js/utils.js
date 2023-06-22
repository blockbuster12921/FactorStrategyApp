"use strict";

// Render dismissable alert
function renderAlert(text, context) {
    var html = `
    <div class="alert alert-` + context + ` alert-dismissible ml-3 my-0 pt-2 pb-1" role="alert">`
    html += text;
    html += `
    <button type="button" class="close pl-0 pr-2 pt-2 pb-0" data-dismiss="alert" aria-label="Close">
        <span>&times;</span>
    </button>
    </div>`;

    return html;
};

// Reset forms when parent modal is hidden
function resetModalForms() {
    $('.modal').on('hidden.bs.modal', function(){
        let form = $(this).find('form')[0];
        if ( form !== undefined ) {
            form.reset();
        }
    });
}

// Initialize selectpicker
function initSelectpicker() {

    $.fn.selectpicker.Constructor.BootstrapVersion = '4';

    // Set selectpicker defaults
    $.fn.selectpicker.Constructor.DEFAULTS.noneSelectedText = "None selected";
    $.fn.selectpicker.Constructor.DEFAULTS.style = "";
    $.fn.selectpicker.Constructor.DEFAULTS.styleBase = "form-control";
}

// Populate a selectpicker select control with months
function populateMonthSelectpicker(selectElement, selectedMonth) {

    let html = '';

    const months = ['January','February','March','April','May','June','July','August','September','October','November','December'];
    months.forEach(function(month, index){
        const monthNumber = index+1;
        html += '<option value="' + monthNumber.toString() + '"';
        if ( (selectedMonth !== null) && (monthNumber === selectedMonth) ) {
            html += ' selected';
        };
        html += '>' + month + '</option>';
    });
    html += '</select>';

    selectElement.html(html).selectpicker('refresh');
};

// Populate a selectpicker select control with years
function populateYearSelectpicker(selectElement, startYear, endYear, selectedYear) {

    let html = '';

    for ( let year=endYear; year>=startYear; year-- ) {
        html += '<option value="' + year.toString() + '"';
        if ( (selectedYear !== null) && (selectedYear === year) ) {
            html += ' selected';
        };
        html += '>' + year.toString() + '</option>';
    };
    html += '</select>';

    selectElement.html(html).selectpicker('refresh');
};