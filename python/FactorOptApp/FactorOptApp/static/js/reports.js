
"use strict";

function makeReportFilename(base, project, context) {
    
    let filename = base + "-";
    filename += project['Name'].replace(/[/\\?%*:|"<>]/g, '_');
    if ( context !== null ) {
        filename += '-' + context;
    }
    filename += '.xlsx';

    return filename;
}

function exportCombinations(project, context, progressElement) {

    // Show progress
    progressElement.prop('hidden', false);

    let xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
      // Do this after transfer complete
      if ( this.readyState == XMLHttpRequest.DONE ) {
          if ( this.status == 200 ) {
            console.log("Download succeeded");

            let blob = null;
            const data = this.response;
            blob = new Blob([data], {type : 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'});

            let objUrl = URL.createObjectURL(blob);

            // Use temporary anchor to download file
            let link = document.createElement('a');
            link.style.display = "none";
            link.href = objUrl;
            link.download = makeReportFilename("Combinations", project, context);;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link); 
          }
          else {
              console.log("Download failed");
          }

          progressElement.prop('hidden', true);
      };
    };

    xhr.open('POST', '/export_project_factor_combinations');
    xhr.responseType = 'blob';
    xhr.setRequestHeader('Content-type', 'application/json');
    xhr.send(JSON.stringify({'ProjectID': project['ID'], 'Context': context}));
};

function exportAnalysis(project, progressElement) {

    // Show progress
    progressElement.prop('hidden', false);

    let xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function () {
        // Do this after transfer complete
        if (this.readyState == XMLHttpRequest.DONE) {
            if (this.status == 200) {
                console.log("Download succeeded");

                let blob = null;
                const data = this.response;
                blob = new Blob([data], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });

                let objUrl = URL.createObjectURL(blob);

                // Use temporary anchor to download file
                let link = document.createElement('a');
                link.style.display = "none";
                link.href = objUrl;
                link.download = makeReportFilename("Analysis", project, null);;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }
            else {
                console.log("Download failed");
            }

            progressElement.prop('hidden', true);
        };
    };

    xhr.open('POST', '/export_project_factor_strategy_analysis');
    xhr.responseType = 'blob';
    xhr.setRequestHeader('Content-type', 'application/json');
    let request = {
        'ProjectID': project['ID'],
    };
    xhr.send(JSON.stringify(request));
};


function exportStockSelection(project, progressElement) {

    // Show progress
    progressElement.prop('hidden', false);

    let xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function () {
        // Do this after transfer complete
        if (this.readyState == XMLHttpRequest.DONE) {
            if (this.status == 200) {
                console.log("Download succeeded");

                let blob = null;
                const data = this.response;
                blob = new Blob([data], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });

                let objUrl = URL.createObjectURL(blob);

                // Use temporary anchor to download file
                let link = document.createElement('a');
                link.style.display = "none";
                link.href = objUrl;
                link.download = makeReportFilename("TickerSelection", project, null);;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }
            else {
                console.log("Download failed");
            }

            progressElement.prop('hidden', true);
        };
    };

    xhr.open('POST', '/export_project_stock_selection');
    xhr.responseType = 'blob';
    xhr.setRequestHeader('Content-type', 'application/json');
    let request = {
        'ProjectID': project['ID'],
    };
    xhr.send(JSON.stringify(request));
};


function exportFactorsDisabled(project, progressElement) {

    // Show progress
    progressElement.prop('hidden', false);

    let xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function () {
        // Do this after transfer complete
        if (this.readyState == XMLHttpRequest.DONE) {
            if (this.status == 200) {
                console.log("Download succeeded");

                let blob = null;
                const data = this.response;
                blob = new Blob([data], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });

                let objUrl = URL.createObjectURL(blob);

                // Use temporary anchor to download file
                let link = document.createElement('a');
                link.style.display = "none";
                link.href = objUrl;
                link.download = makeReportFilename("FactorsDisabled", project, null);;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }
            else {
                console.log("Download failed");
            }

            progressElement.prop('hidden', true);
        };
    };

    xhr.open('POST', '/export_project_factors_disabled');
    xhr.responseType = 'blob';
    xhr.setRequestHeader('Content-type', 'application/json');
    let request = {
        'ProjectID': project['ID'],
    };
    xhr.send(JSON.stringify(request));
};


function exportStockRanking(project, progressElement) {

    // Show progress
    progressElement.prop('hidden', false);

    let xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function () {
        // Do this after transfer complete
        if (this.readyState == XMLHttpRequest.DONE) {
            if (this.status == 200) {
                console.log("Download succeeded");

                let blob = null;
                const data = this.response;
                blob = new Blob([data], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });

                let objUrl = URL.createObjectURL(blob);

                // Use temporary anchor to download file
                let link = document.createElement('a');
                link.style.display = "none";
                link.href = objUrl;
                link.download = makeReportFilename("StockRanking", project, null);;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }
            else {
                console.log("Download failed");
            }

            progressElement.prop('hidden', true);
        };
    };

    xhr.open('POST', '/export_project_stock_ranking');
    xhr.responseType = 'blob';
    xhr.setRequestHeader('Content-type', 'application/json');
    let request = {
        'ProjectID': project['ID'],
        'StrategyID': $('#stock-ranking-report-strategy-select').val(),
    };
    console.log(request);
    xhr.send(JSON.stringify(request));
};


function exportProjectGroupStockRanking(projectGroup, progressElement) {

    // Show progress
    progressElement.prop('hidden', false);

    let xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function () {
        // Do this after transfer complete
        if (this.readyState == XMLHttpRequest.DONE) {
            if (this.status == 200) {
                console.log("Download succeeded");

                let blob = null;
                const data = this.response;
                blob = new Blob([data], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });

                let objUrl = URL.createObjectURL(blob);

                // Use temporary anchor to download file
                let link = document.createElement('a');
                link.style.display = "none";
                link.href = objUrl;
                link.download = makeReportFilename("StockRanking", projectGroup, null);;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }
            else {
                console.log("Download failed");
                console.log(this.responseText);
                $('#project-group-report-error-detail').text(this.responseText);
                $('#project-group-report-error-modal').modal('show');
            }

            progressElement.prop('hidden', true);
        } else if(this.readyState == XMLHttpRequest.HEADERS_RECEIVED) {
            if(this.status == 200) {
                this.responseType = "blob";
            } else {
                this.responseType = "text";
            }
        }
    };

    xhr.open('POST', '/export_project_group_stock_ranking');
    xhr.setRequestHeader('Content-type', 'application/json');
    let request = {
        'ProjectGroupID': projectGroup['ID'],
    };
    xhr.send(JSON.stringify(request));
};


function exportDriverParamSelection(project, progressElement) {

    // Show progress
    progressElement.prop('hidden', false);

    let xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function () {
        // Do this after transfer complete
        if (this.readyState == XMLHttpRequest.DONE) {
            if (this.status == 200) {
                console.log("Download succeeded");

                let blob = null;
                const data = this.response;
                blob = new Blob([data], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });

                let objUrl = URL.createObjectURL(blob);

                // Use temporary anchor to download file
                let link = document.createElement('a');
                link.style.display = "none";
                link.href = objUrl;
                link.download = makeReportFilename("OutlierSelection", project, null);;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }
            else {
                console.log("Download failed");
            }

            progressElement.prop('hidden', true);
        };
    };

    xhr.open('POST', '/export_project_driver_param_selection');
    xhr.responseType = 'blob';
    xhr.setRequestHeader('Content-type', 'application/json');
    let request = {
        'ProjectID': project['ID'],
    };
    xhr.send(JSON.stringify(request));
};


function exportFactorFilter(project, progressElement) {

    // Show progress
    progressElement.prop('hidden', false);

    let xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function () {
        // Do this after transfer complete
        if (this.readyState == XMLHttpRequest.DONE) {
            if (this.status == 200) {
                console.log("Download succeeded");

                let blob = null;
                const data = this.response;
                blob = new Blob([data], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });

                let objUrl = URL.createObjectURL(blob);

                // Use temporary anchor to download file
                let link = document.createElement('a');
                link.style.display = "none";
                link.href = objUrl;
                link.download = makeReportFilename("FactorFilter", project, null);;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }
            else {
                console.log("Download failed");
            }

            progressElement.prop('hidden', true);
        };
    };

    xhr.open('POST', '/export_project_factor_filter');
    xhr.responseType = 'blob';
    xhr.setRequestHeader('Content-type', 'application/json');
    let request = {
        'ProjectID': project['ID'],
    };
    xhr.send(JSON.stringify(request));
};


function exportFactorClusters(project, progressElement) {

    // Show progress
    progressElement.prop('hidden', false);

    let xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function () {
        // Do this after transfer complete
        if (this.readyState == XMLHttpRequest.DONE) {
            if (this.status == 200) {
                console.log("Download succeeded");

                let blob = null;
                const data = this.response;
                blob = new Blob([data], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });

                let objUrl = URL.createObjectURL(blob);

                // Use temporary anchor to download file
                let link = document.createElement('a');
                link.style.display = "none";
                link.href = objUrl;
                link.download = makeReportFilename("FactorClusters", project, null);;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }
            else {
                console.log("Download failed");
            }

            progressElement.prop('hidden', true);
        };
    };

    xhr.open('POST', '/export_project_factor_clusters');
    xhr.responseType = 'blob';
    xhr.setRequestHeader('Content-type', 'application/json');
    let request = {
        'ProjectID': project['ID'],
    };
    xhr.send(JSON.stringify(request));
};


function exportDriverFactorWeights(project, progressElement) {

    // Show progress
    progressElement.prop('hidden', false);

    let xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function () {
        // Do this after transfer complete
        if (this.readyState == XMLHttpRequest.DONE) {
            if (this.status == 200) {
                console.log("Download succeeded");

                let blob = null;
                const data = this.response;
                blob = new Blob([data], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });

                let objUrl = URL.createObjectURL(blob);

                // Use temporary anchor to download file
                let link = document.createElement('a');
                link.style.display = "none";
                link.href = objUrl;
                link.download = makeReportFilename("FactorCorrelation", project, null);;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }
            else {
                console.log("Download failed");
            }

            progressElement.prop('hidden', true);
        };
    };

    xhr.open('POST', '/export_project_driver_factor_weights');
    xhr.responseType = 'blob';
    xhr.setRequestHeader('Content-type', 'application/json');
    let request = {
        'ProjectID': project['ID'],
    };
    xhr.send(JSON.stringify(request));
};
