# Routes and views for the flask application.

from datetime import datetime, date
import time
from flask import render_template, request, redirect, url_for, send_file, jsonify, Response, make_response, flash
from flask.json import JSONEncoder
from bson.objectid import ObjectId
import json
from io import BytesIO
import pandas as pd
from . import app
from . import forms
from FactorCore import cache, database, stock_data, factor_calc, generate, optimize, settings, analysis, data_io, stock_selection, stock_ranking, driver, factor_filter, factor_disable, factor_cluster
from FactorCore.run_project import ProjectRunner
from FactorCore.utils import MeasureTime, MeasureBlockTime

class CustomJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, ObjectId):
            return None
        return JSONEncoder.default(self, obj)
app.json_encoder = CustomJSONEncoder

app_title = 'FactorApp'

# Database
db = database.DB()

# Home page
@app.route('/', methods=["GET"])
@app.route('/home', methods=["GET"])
def home():

    form_dict = {}    
    form_dict['AddProject'] = forms.AddProjectForm()

    return render_template('index.html', title=app_title, forms=form_dict)

@app.route('/datasets', methods=["GET"])
def datasets():

    return render_template('datasets.html', title=app_title)

@app.route('/project_groups', methods=["GET"])
def project_groups():

    return render_template('project_groups.html', title=app_title)

@app.route('/get_datasets', methods=["GET"])
def get_datasets():

    datasets = db.get_datasets()

    for dataset in datasets:
        dataset['URL'] = url_for('view_dataset', dataset_id=dataset['ID'])
        dataset['Region'] = db.get_region_name(dataset['RegionID'])
        dataset['Sector'] = db.get_sector_name(dataset['SectorID'])
        dataset['DataStartDate'] = None
        dataset['DataEndDate'] = None
        dataset['StockCount'] = None
        dataset['FactorCount'] = None
        dataset['DataFilename'] = None
        dataset['DataLoadedTimestamp'] = None

        data_info = db.get_dataset_data_info(dataset['ID'])
        if data_info is not None:
            if len(data_info['Dates']) > 0:
                dataset['DataStartDate'] = data_info['Dates'][0]
                dataset['DataEndDate'] = data_info['Dates'][-1]
            dataset['StockCount'] = len(data_info['Stocks'])
            dataset['FactorCount'] = len(data_info['Factors'])
            dataset['DataFilename'] = data_info['Filename']
            dataset['DataLoadedTimestamp'] = data_info['Created']

    return jsonify(datasets)

@app.route('/create_dataset', methods=["POST"])
def create_dataset():

    request_json = request.get_json()

    try:
        dataset = db.create_dataset()
    except:
        return ('Failed to create dataset', 400)

    response = { 'DatasetID': dataset['ID'], 'DatasetURL': url_for('view_dataset', dataset_id=dataset['ID']) }

    return make_response(jsonify(response), 200)

@app.route('/delete_dataset', methods=["POST"])
def delete_dataset():

    request_json = request.get_json()
    dataset_id = request_json['DatasetID']

    try:
        project = db.delete_dataset(dataset_id)
    except Exception as e:
        return ('Failed to delete dataset', 400)

    return jsonify("Dataset deleted successfully")

@app.route('/datasets/dataset/<dataset_id>', methods=["GET"])
def view_dataset(dataset_id):

    dataset = db.get_dataset(dataset_id)
    if dataset is None:
        flash("Dataset {} not found".format(dataset_id), "error")
        return redirect(url_for('datasets'))

    reference_data = db.get_reference_data()

    return render_template('dataset.html', title=dataset['Name'], dataset=dataset, reference_data=reference_data)

@app.route('/update_dataset_metadata', methods=["PUT"])
def update_dataset_metadata():

    request_json = request.get_json()

    try:
        db.update_dataset_metadata(request_json)
    except Exception as e:
        app.logger.exception("Failed to update dataset metadata")
        return (str(e), 400)

    return jsonify("Dataset metadata updated")

@app.route('/load_dataset_data', methods=["POST"])
def load_dataset_data():

    try:
        uploaded_file = request.files['file']
        dataset_id = str(request.form['datasetId'])

        # Load data
        loader = data_io.DataLoader()
        loader.load_from_excel(uploaded_file)

        # Save data to db
        db.set_dataset_data(dataset_id, uploaded_file.filename, loader.dates, loader.stocks, loader.factor_names, loader.returns_df, loader.factors_df)

        # Construct response
        response = { 
            'FactorCount': len(loader.factor_names),
            'StockCount': len(loader.stocks),
            }

    except Exception as e:
        app.logger.exception("Failed to load data from uploaded file")
        return (str(e), 400)

    return jsonify(response)

@app.route('/get_dataset_data_info', methods=["POST"])
def get_dataset_data_info():

    request_json = request.get_json()
    dataset_id = request_json['DatasetID']

    response = db.get_dataset_data_info(dataset_id)

    return jsonify(response)

@app.route('/get_project_groups', methods=["GET"])
def get_project_groups():

    project_groups = db.get_project_groups()

    for group in project_groups:
        group['URL'] = url_for('view_project_group', project_group_id=group['ID'])

    return jsonify(project_groups)

@app.route('/create_project_group', methods=["POST"])
def create_project_group():

    request_json = request.get_json()

    project_ids = request_json['ProjectIDs']

    try:
        project_group = db.create_project_group(project_ids)
    except:
        return ('Failed to create project group', 400)

    response = { 'ProjectGroupID': project_group['ID'], 'ProjectGroupURL': url_for('view_project_group', project_group_id=project_group['ID']) }

    return make_response(jsonify(response), 200)

@app.route('/project_groups/project_group/<project_group_id>', methods=["GET"])
def view_project_group(project_group_id):

    project_group = db.get_project_group(project_group_id)
    if project_group is None:
        flash("Project Group {} not found".format(project_group_id), "error")
        return redirect(url_for('project_groups'))

    factor_strategies = db.get_factor_strategies()

    return render_template('project_group.html', title=project_group['Name'], project_group=project_group, factor_strategies=factor_strategies)

@app.route('/delete_project_group', methods=["POST"])
def delete_project_group():

    request_json = request.get_json()
    group_id = request_json['ProjectGroupID']

    try:
        project = db.delete_project_group(group_id)
    except Exception as e:
        return ('Failed to delete project group', 400)

    return jsonify("Project Group deleted successfully")

@app.route('/update_project_group_metadata', methods=["PUT"])
def update_project_group_metadata():

    request_json = request.get_json()

    try:
        db.update_project_group_metadata(request_json)
    except Exception as e:
        app.logger.exception("Failed to update project group metadata")
        return (str(e), 400)

    return jsonify("Project group metadata updated")

@app.route('/projects', methods=["GET"])
def projects():

    form_dict = {}    
    form_dict['AddProject'] = forms.AddProjectForm()

    return render_template('projects.html', title=app_title, forms=form_dict)

@app.route('/get_projects', methods=["POST"])
def get_projects():

    request_json = request.get_json()
    project_ids = request_json.get('ProjectIDs')
    get_data_info = request_json.get('DataInfo', False)

    projects = db.get_projects(project_ids=project_ids, include_deleted=True)

    dataset_names = { dataset['ID']: dataset['Name'] for dataset in db.get_datasets() }

    for project in projects:
        project['URL'] = url_for('view_project', project_id=project['ID'])

        project_dataset_names = [dataset_names.get(dataset_id, "DELETED") for dataset_id in project['DatasetIDs']]
        project['DatasetNames'] = "; ".join(project_dataset_names)

        if get_data_info:
            data_info = db.get_project_data_info(project['ID'])
            if data_info is not None:
                if len(data_info['Dates']) > 0:
                    project['DataStartDate'] = data_info['Dates'][0]
                    project['DataEndDate'] = data_info['Dates'][-1]
                project['StockCount'] = len(data_info['Stocks'])

    return jsonify(projects)

@app.route('/add_project', methods=["POST"])
def add_project():

    form = forms.AddProjectForm()
    name = form.new_name.data

    if db.existing_project_has_name(name):
        app.logger.error("A project with the name '{}' already exists".format(name))
        flash("A project with the name '{}' already exists".format(name), "error")
        return redirect(url_for('home'))

    try:
        project = db.create_project(name)
    except:
        app.logger.error("Failed to create project")
        flash("Failed to add Project", "error")
        return redirect(url_for('projects'))

    return redirect(url_for('view_project', project_id=project['ID']))

@app.route('/clone_project', methods=["POST"])
def clone_project():

    request_json = request.get_json()
    project_id = request_json['ProjectID']

    try:
        project = db.clone_project(project_id)
    except Exception as e:
        app.logger.error("Failed to clone project")
        flash("Failed to duplicate Project", "error")
        return redirect(url_for('projects'))

    response = { 'ProjectID': project['ID'], 'ProjectURL': url_for('view_project', project_id=project['ID']) }

    return jsonify(response)

@app.route('/delete_project', methods=["POST"])
def delete_project():

    request_json = request.get_json()
    project_id = request_json['ProjectID']

    try:
        project = db.delete_project(project_id)
    except Exception as e:
        return ('Failed to delete project', 400)

    return jsonify("Project deleted successfully")

@app.route('/archive_project', methods=["POST"])
def archive_project():

    request_json = request.get_json()
    project_id = request_json['ProjectID']

    try:
        project = db.archive_project(project_id)
    except:
        return ('Failed to archive project', 400)

    return jsonify("Project archived successfully")

@app.route('/restore_project', methods=["POST"])
def restore_project():

    request_json = request.get_json()
    project_id = request_json['ProjectID']

    try:
        project = db.restore_project(project_id)
    except:
        return ('Failed to restore project', 400)

    return jsonify("Project restored successfully")

@app.route('/projects/project/<project_id>', methods=["GET"])
def view_project(project_id):

    project = db.get_project(project_id)
    if project is None:
        flash("Project {} not found".format(project_id), "error")
        return redirect(url_for('home'))

    if 'OOSEndDate' not in project:
        project['OOSEndDate'] = None 

    if 'DatasetIDs' not in project:
        project['DatasetIDs'] = [] 

    datasets = db.get_datasets()
    factor_strategies = db.get_factor_strategies()

    return render_template('project.html', title=project['Name'], project=project, datasets=datasets, factor_strategies=factor_strategies)

@app.route('/update_project_metadata', methods=["PUT"])
def update_project_metadata():

    request_json = request.get_json()

    try:
        db.update_project_metadata(request_json)
    except Exception as e:
        app.logger.exception("Failed to update project metadata")
        return (str(e), 400)

    return jsonify("Project metadata updated")

@app.route('/clear_project_data', methods=["POST"])
def clear_project_data():

    request_json = request.get_json()
    project_id = request_json['ProjectID']

    db.clear_project_data(project_id)

    # TODO: clear caches

    return jsonify("Data cleared")

@app.route('/get_project_settings', methods=["POST"])
def get_project_settings():

    request_json = request.get_json()
    project_id = request_json['ProjectID']

    project_settings = db.get_project_settings(project_id)
    if project_settings is None:
        project_settings = {}

    project_settings = settings.overlay_default_project_settings(project_settings)

    return jsonify(project_settings)

@app.route('/set_project_settings', methods=["POST"])
def set_project_settings():

    request_json = request.get_json()
    project_id = request_json['ProjectID']

    db.set_project_settings(project_id, request_json['Settings'])

    return jsonify("Success")

@app.route('/clear_project_settings', methods=["POST"])
def clear_project_settings():

    request_json = request.get_json()
    project_id = request_json['ProjectID']

    db.clear_project_settings(project_id)

    return jsonify("Success")

@app.route('/get_project_stocks_info', methods=["POST"])
def get_project_stocks_info():

    request_json = request.get_json()
    project_id = request_json['ProjectID']

    data_info = db.get_project_data_info(project_id)

    if data_info is None:
        return jsonify(None)

    dataset_names = { dataset['ID']: dataset['Name'] for dataset in db.get_datasets() }

    response = { 'Stocks': data_info['Stocks'] }
    for index, stock in enumerate(response['Stocks']):
        stock['Index'] = index
        stock['Enabled'] = True
        stock['ForwardMonthEnabled'] = True
        stock['DatasetName'] = dataset_names[stock['DatasetID']]

    disabled = db.get_project_stocks_disabled(project_id)
    if disabled is not None:
        for item in disabled:
            for stock in response['Stocks']:
                if (stock['DatasetID'] == item['DatasetID']) and (stock['DatasetIndex'] == item['DatasetIndex']):
                    stock['Enabled'] = False
                    break

    disabled = db.get_project_stocks_forward_month_disabled(project_id)
    if disabled is not None:
        for item in disabled:
            for stock in response['Stocks']:
                if (stock['DatasetID'] == item['DatasetID']) and (stock['DatasetIndex'] == item['DatasetIndex']):
                    stock['ForwardMonthEnabled'] = False
                    break

    return jsonify(response)

@app.route('/set_project_stocks_disabled_status', methods=["POST"])
def set_project_stocks_disabled_status():

    request_json = request.get_json()

    db.set_project_stocks_disabled(request_json['ProjectID'], request_json['StocksDisabled'])

    return jsonify("Success")

@app.route('/set_project_stocks_forward_month_disabled_status', methods=["POST"])
def set_project_stocks_forward_month_disabled_status():

    request_json = request.get_json()

    db.set_project_stocks_forward_month_disabled(request_json['ProjectID'], request_json['StocksDisabled'])

    return jsonify("Success")

@app.route('/get_project_factors_info', methods=["POST"])
def get_project_factors_info():

    request_json = request.get_json()
    project_id = request_json['ProjectID']

    data_info = db.get_project_data_info(project_id)
    if data_info is None:
        return jsonify(None)

    response = { 'Factors': [{'Name': factor, 'Enabled': True} for factor in data_info['Factors']] }

    disabled = db.get_project_factors_disabled(project_id)
    if disabled is not None:
        for disabled_factor in disabled:
            for f in response['Factors']:
                if f['Name'] == disabled_factor:
                    f['Enabled'] = False
                    break

    return jsonify(response)

@app.route('/set_project_factors_disabled_status', methods=["POST"])
def set_project_factors_disabled_status():

    request_json = request.get_json()

    db.set_project_factors_disabled(request_json['ProjectID'], request_json['FactorsDisabled'])

    return jsonify("Success")

@app.route('/get_project_run_state', methods=["PUT"])
def get_project_run_state():

    request_json = request.get_json()
    project_id = request_json['ProjectID']

    response = db.get_project_run_state(project_id)

    return jsonify(response)

@app.route('/get_project_group_run_state', methods=["PUT"])
def get_project_group_run_state():

    request_json = request.get_json()
    project_group_id = request_json['ProjectGroupID']

    project_group = db.get_project_group(project_group_id)

    response = []
    for project_id in project_group['ProjectIDs']:
        project = db.get_project(project_id)
        response.append({ 
            'ProjectID': project_id,
            'ProjectName': project['Name'],
            'ProjectRunState': db.get_project_run_state(project_id)
        })

    return jsonify(response)

@app.route('/run_project', methods=["POST"])
def run_project():

    request_json = request.get_json()
    project_id = request_json['ProjectID']

    try:
        runner = ProjectRunner(db, project_id)
        run_complete = runner.run()
    except Exception as e:
        app.logger.exception("Failed to run project")
        return (str(e), 400)

    response = db.get_project_run_state(project_id)

    return jsonify(response)

@app.route('/run_project_group', methods=["POST"])
def run_project_group():

    request_json = request.get_json()
    project_group_id = request_json['ProjectGroupID']

    project_group = db.get_project_group(project_group_id)

    next_incomplete_project_id = None
    for project_id in project_group['ProjectIDs']:
        run_state = db.get_project_run_state(project_id)
        if (run_state is None) or (not run_state['RunComplete']):
            next_incomplete_project_id = project_id
            break

    if next_incomplete_project_id is None:
        response = { 'RunComplete': True }
        return jsonify(response)

    try:
        runner = ProjectRunner(db, next_incomplete_project_id)
        runner.run()
    except Exception as e:
        app.logger.exception("Failed to run project {}".format(next_incomplete_project_id))
        return (str(e), 400)

    response = { 'RunComplete': False }
    return jsonify(response)

@app.route('/reset_project_run', methods=["POST"])
def reset_project_run():

    request_json = request.get_json()
    project_id = request_json['ProjectID']

    db.clear_project_run_data(project_id)

    return jsonify(True)

@app.route('/reset_project_group_run', methods=["POST"])
def reset_project_group_run():

    request_json = request.get_json()
    project_group_id = request_json['ProjectGroupID']

    project_group = db.get_project_group(project_group_id)

    for project_id in project_group['ProjectIDs']:
        db.clear_project_run_data(project_id)

    return jsonify(True)

@app.route('/export_project_factor_strategy_analysis', methods=["POST"])
def export_project_factor_strategy_analysis():

    request_json = request.get_json()

    try:
        project_id = request_json['ProjectID']

        results, pairs_range = analysis.collate_factor_strategy_results(db, project_id)

        data_info = db.get_project_data_info(project_id)

        wb = analysis.generate_excel_report(
            results, data_info['Factors'], pairs_range,
            settings.overlay_default_project_settings(db.get_project_settings(project_id))
            )

        buffer = BytesIO()
        wb.save(buffer)
        buffer.seek(0)

        xlsx_mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        response = send_file(buffer, as_attachment=True, attachment_filename='Analysis.xlsx', mimetype=xlsx_mimetype)

    except Exception as e:
        app.logger.exception("Failed to run export_project_factor_strategy_analysis")
        return (str(e), 400)

    return response

@app.route('/get_project_factor_strategy_analysis', methods=["POST"])
def get_project_factor_strategy_analysis():

    request_json = request.get_json()

    try:
        project_id = request_json['ProjectID']

        results, pairs_range = analysis.collate_factor_strategy_results(db, project_id)

        response = {
            'Results': results,
            'PairsRange': { 'start': pairs_range[0], 'end': pairs_range[1] },
        }

    except Exception as e:
        app.logger.exception("Failed to run export_project_factor_strategy_analysis")
        return (str(e), 400)

    return response

@app.route('/export_project_factor_combinations', methods=["POST"])
def export_project_factor_combinations():

    request_json = request.get_json()
    
    try:
        project_id = request_json['ProjectID']
        context = request_json['Context']

        data_info = db.get_project_data_info(project_id)
        factor_names = data_info['Factors']
        factor_count = len(factor_names)

        combinations = db.get_project_factor_combinations(project_id, target_month=None, context=context)

        cluster_factors_by_month = {}

        comb_list = []
        for comb in combinations:

            cluster_factors = cluster_factors_by_month.get(comb['TargetMonth'])
            if cluster_factors is None:
                cluster_df = db.get_project_factor_clusters(project_id, comb['TargetMonth'])
                cluster_factors = factor_cluster.get_clustered_factors_from_df(cluster_df)
                cluster_factors_by_month[comb['TargetMonth']] = cluster_factors

            factors_str = ""
            for cluster in comb['Clusters']:
                if cluster < 0:
                    factors_str += "-"
                    cluster = -cluster
                else:
                    factors_str += '+'
                factors_str += "[{}]".format(",".join([factor_names[f] for f in cluster_factors[cluster]]))

            row_data = [
                factors_str,
                len(comb['Clusters']),
                comb['TargetMonth'],
                float(comb['Score']),
                comb['Context'].capitalize(),
            ]

            comb_list.append(row_data)

        df = pd.DataFrame(columns=['Factors','# Clusters','Target Month','Score','Source'], data=comb_list)

        # Construct Excel response
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter', datetime_format='mmm-yy') as writer:
            df.to_excel(writer, sheet_name='Combinations', index=False)
            worksheet = writer.sheets['Combinations']
            worksheet.set_column('A:A', 60)
            worksheet.set_column('B:B', 10)
            worksheet.set_column('C:C', 14)
            worksheet.set_column('D:D', 14)
            for col_index, col_name in enumerate(df.columns.values):
                worksheet.write(0, col_index, col_name)
            worksheet.autofilter(0, 0, 1+len(comb_list), len(df.columns)-1)
        buffer.seek(0)
        xlsx_mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        response = send_file(buffer, as_attachment=True, attachment_filename='Combinations.xlsx', mimetype=xlsx_mimetype)

    except Exception as e:
        app.logger.exception("Failed to export combinations")
        return (str(e), 400)

    return response

@app.route('/export_project_stock_selection', methods=["POST"])
def export_project_stock_selection():

    request_json = request.get_json()
    
    try:
        project_id = request_json['ProjectID']

        generator = stock_selection.StockSelectionReportGenerator(db, project_id)
        wb = generator.generate()

        buffer = BytesIO()
        wb.save(buffer)
        buffer.seek(0)

        xlsx_mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        response = send_file(buffer, as_attachment=True, attachment_filename='TickerSelection.xlsx', mimetype=xlsx_mimetype)

    except Exception as e:
        app.logger.exception("Failed to export stock selection report")
        return (str(e), 400)

    return response

@app.route('/export_project_factors_disabled', methods=["POST"])
def export_project_factors_disabled():

    request_json = request.get_json()
    
    try:
        project_id = request_json['ProjectID']

        generator = factor_disable.FactorsDisabledReportGenerator(db, project_id)
        wb = generator.generate()

        buffer = BytesIO()
        wb.save(buffer)
        buffer.seek(0)

        xlsx_mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        response = send_file(buffer, as_attachment=True, attachment_filename='FactorsDisabled.xlsx', mimetype=xlsx_mimetype)

    except Exception as e:
        app.logger.exception("Failed to export factors disabled report")
        return (str(e), 400)

    return response

@app.route('/export_project_driver_param_selection', methods=["POST"])
def export_project_driver_param_selection():

    request_json = request.get_json()
    
    try:
        project_id = request_json['ProjectID']

        generator = driver.DriverParamSelectionReportGenerator(db, project_id)
        wb = generator.generate(['FactorOutlierRejection','ReturnsOutlierRejection'])

        buffer = BytesIO()
        wb.save(buffer)
        buffer.seek(0)

        xlsx_mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        response = send_file(buffer, as_attachment=True, attachment_filename='ParameterSelection.xlsx', mimetype=xlsx_mimetype)

    except Exception as e:
        app.logger.exception("Failed to export stock selection report")
        return (str(e), 400)

    return response

@app.route('/export_project_stock_ranking', methods=["POST"])
def export_project_stock_ranking():

    request_json = request.get_json()
    
    try:
        project_id = request_json['ProjectID']
        strategy_id = request_json['StrategyID']

        generator = stock_ranking.StockRankingReportGenerator(db, strategy_id)
        wb = generator.generate_for_project(project_id)

        buffer = BytesIO()
        wb.save(buffer)
        buffer.seek(0)

        xlsx_mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        response = send_file(buffer, as_attachment=True, attachment_filename='StockRanking.xlsx', mimetype=xlsx_mimetype)

    except Exception as e:
        app.logger.exception("Failed to export stock ranking report")
        return (str(e), 400)

    return response

@app.route('/export_project_group_stock_ranking', methods=["POST"])
def export_project_group_stock_ranking():

    request_json = request.get_json()
    
    try:
        project_group_id = request_json['ProjectGroupID']

        generator = stock_ranking.StockRankingReportGenerator(db, strategy_id=None)
        wb = generator.generate_for_project_group(project_group_id)

        buffer = BytesIO()
        wb.save(buffer)
        buffer.seek(0)

        xlsx_mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        response = send_file(buffer, as_attachment=True, attachment_filename='StockRanking.xlsx', mimetype=xlsx_mimetype)

    except Exception as e:
        app.logger.exception("Failed to export stock ranking report")
        return (str(e), 400)

    return response

@app.route('/export_project_factor_filter', methods=["POST"])
def export_project_factor_filter():

    request_json = request.get_json()
    
    try:
        project_id = request_json['ProjectID']

        generator = factor_filter.FactorFilterReportGenerator(db, project_id)
        wb = generator.generate()

        buffer = BytesIO()
        wb.save(buffer)
        buffer.seek(0)

        xlsx_mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        response = send_file(buffer, as_attachment=True, attachment_filename='FactorFilter.xlsx', mimetype=xlsx_mimetype)

    except Exception as e:
        app.logger.exception("Failed to export factor filter report")
        return (str(e), 400)

    return response

@app.route('/export_project_factor_clusters', methods=["POST"])
def export_project_factor_clusters():

    request_json = request.get_json()
    
    try:
        project_id = request_json['ProjectID']

        generator = factor_cluster.FactorClusterReportGenerator(db, project_id)
        wb = generator.generate()

        buffer = BytesIO()
        wb.save(buffer)
        buffer.seek(0)

        xlsx_mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        response = send_file(buffer, as_attachment=True, attachment_filename='FactorClusters.xlsx', mimetype=xlsx_mimetype)

    except Exception as e:
        app.logger.exception("Failed to export factor clusters report")
        return (str(e), 400)

    return response

@app.route('/export_project_driver_factor_weights', methods=["POST"])
def export_project_driver_factor_weights():

    request_json = request.get_json()
    
    try:
        project_id = request_json['ProjectID']

        generator = driver.FactorWeightsReportGenerator(db, project_id)
        wb = generator.generate()

        buffer = BytesIO()
        wb.save(buffer)
        buffer.seek(0)

        xlsx_mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        response = send_file(buffer, as_attachment=True, attachment_filename='FactorWeights.xlsx', mimetype=xlsx_mimetype)

    except Exception as e:
        app.logger.exception("Failed to export driver factor weights report")
        return (str(e), 400)

    return response