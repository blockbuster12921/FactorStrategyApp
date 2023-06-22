"use strict";

function dtRenderHyperlink(data, type, row, meta, url, openInNewWindow) {
    if (type !== 'display' ) {
        return data;
    }
    if ( data === null ) {
        return "";
    }
    let html = '<a href="' + url + '"';
    if ( openInNewWindow ) {
        html += ' target="_blank"';
    }
    html += '>' + data + '</a>';

    return html;
}

function dtRenderDatetime(data, type, row, meta, format) {
    if (type !== 'display') {
        return data
    }

    if ( (data === null) || (data === undefined) ) {
        return ""
    }
    return moment.utc(data).local().format(format)
}

function dtRenderMonth(data, type, row, meta) {
    return dtRenderDatetime(data, type, row, meta, "MMM Y")
}

function dtRenderTimestamp(data, type, row, meta) {
    return dtRenderDatetime(data, type, row, meta, "YYYY-MM-DD HH:mm:ss.SS")
}

const dtControlColumnDef = { data: null, className: "control", defaultContent: "", orderable: false, searchable: false };

function dtChildRowRenderer( api, rowIdx, columns ) {

    let html = '<dl class="row pl-2">'
    for ( var i=0; i<columns.length; i++ ) {
        const col = columns[i];
        if ( col.hidden ) {
            html += '<dt class="col-xl-2 col-lg-3 col-md-4 col-sm-4 col-6 mt-1 mb-0">' + col.title + ': </dt>';
            html += '<dd class="col-xl-10 col-lg-9 col-md-8 col-sm-8 col-6 mt-1 mb-0 align-self-end">' + col.data + '</dd>';
        }
    }
    html += '</dl>';
    return html;
}

function initDataTable(table_id, data, columns, label, order, filter_id)
{
    let domFilter = filter_id === "" ? "f" : "";
    let responsiveType = (columns[0]['className'] === "control") ? 'column' : 'inline';

    const minLength = 10;
 
    $('#'+table_id).DataTable( 
    {
        data: data,
        columns: columns,
        order: order,
        responsive: {
            details: {
                type: responsiveType,
                renderer: dtChildRowRenderer,
            }
        },
        dom: "<'d-flex flex-wrap justify-content-between'<'pt-1 pl-3 pr-1'l><'float-left pt-1 pl-3 pr-3'"+domFilter+">>" +
             "<'row'<'col-sm-12'tr>>" +
             "<'d-flex flex-wrap justify-content-between'<'py-1 pl-3 pr-1 mt-1'p><'float-left py-1 pl-3 pr-3'i>>",
        "info": false,
        "lengthChange": true,
        "searching": true,
        "pagingType": "simple_numbers",
        "stateSave": true,
        "pageLength": 10,
        "lengthMenu": [ minLength, 20, 50, 100 ],
        "language": { "search": "<i class='fas fa-search' aria-hidden='true'></i>",
                    "lengthMenu": "Show _MENU_ " + label,
                    "info": "Showing _START_ to _END_ of _TOTAL_ " + label,
                    "infoFiltered":   "(filtered from _MAX_ total " + label + ")",
                    "zeroRecords": "No " + label + " found",
                    "paginate": { "previous": '<i class="fas fa-chevron-left"></i>', "next": '<i class="fas fa-chevron-right"></i>' },
        },
        fnDrawCallback: function(oSettings) {
            if( oSettings.aoData.length <= oSettings._iDisplayLength ){
                $(oSettings.nTableWrapper).find('.dataTables_paginate').hide();
            }
            else {
                $(oSettings.nTableWrapper).find('.dataTables_paginate').show();
            }
            if( oSettings.aoData.length <= minLength ){
                $(oSettings.nTableWrapper).find('.dataTables_length').hide();
            }
            else {
                $(oSettings.nTableWrapper).find('.dataTables_length').show();
            }
        },
    });

    if ( filter_id ) {
        $('#'+filter_id).keyup(function() {
            $('#'+table_id).DataTable().search($(this).val()).draw();
        });
    }
}

function clearDataTable(table_id)
{
    if ( $.fn.dataTable.isDataTable('#'+table_id) ) {
        let table = $('#'+table_id).DataTable();
        table.clear();
        table.columns.adjust().responsive.recalc();
        table.draw();
    }
}

function updateDataTable(table_id, data)
{
    let table = $('#'+table_id).DataTable();
    table.clear();
    table.rows.add(data);
    table.columns.adjust().responsive.recalc();
    table.draw();
}
