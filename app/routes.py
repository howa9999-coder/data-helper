from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import os
from werkzeug.utils import secure_filename
import re
from flask import send_file
import io, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import missingno as msno
from app.functions import clear_session, handle_file_upload, clear_upload_folder, new_column_match_keywords, filter_rows_by_dual_keywords, extract_keywords, articles_per_period
from app import app



# Define the folder where uploaded files will be stored
UPLOAD_FOLDER = 'uploads'

# Create the uploads folder if it doesn’t already exist (so no error on first run)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Define a Flask route '/practice' that handles both GET and POST requests
@app.route('/')
def home():
    return render_template('home.htm', title= 'DataHelper', this_is_home = True)

import os
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')


@app.route('/search', methods=['GET', 'POST'])
def search():
    # Initialize variables for template rendering
    tables = new_table = columns = shape_rows = shape_cols = filename = None

    # ----------- FORM 1: Handle file upload -----------
    if request.method == 'POST' and 'upload' in request.form:
        session_list= ['selected_columns', 'new_column', 'keywords', 'no_result', 'regex_option', 'case_sensitive', 'modified_df']    
        clear_session(session_list)
        tables, columns, shape, filename = handle_file_upload(request, UPLOAD_FOLDER)
        shape_rows = shape[0]
        shape_cols = shape[1]

    # ----------- FORM 2: Handle stats request -----------
    elif request.method == 'POST' and 'stats' in request.form:
        selected_column = request.form.get('column')
        new_column = request.form.get('column_name')
        keywords = [k.strip() for k in request.form.get('keywords', '').split(',')]
        no_result = request.form.get('no_match_value')
        regex_option = request.form.get('regex')
        sensitive = request.form.get('case_sensitive') == 'true'
        filename = session.get('uploaded_file_name')        
        # Save all selections in session
        session['selected_column'] = selected_column
        session['new_column'] = new_column
        session['keywords'] = keywords
        session['no_result'] = no_result
        session['regex_option'] = regex_option
        session['case_sensitive'] = sensitive
        filepath = session.get('data_file')
        if filepath and selected_column:
            df_a = pd.read_excel(filepath)
            df = new_column_match_keywords(
                df_a, selected_column, new_column, keywords,
                no_match_keyword=no_result, regex=regex_option,
                case_sensitive=sensitive
            )
            shape = df.shape
            shape_rows = shape[0]
            shape_cols = shape[1]
            new_table = df.to_html(classes='table table-bordered table-striped table-hover align-middle', index=False, border=0)
            session['modified_df'] = df.to_csv(index=False)

    # Retrieve columns from session if not set (for dropdown persistence)
    if columns is None and session.get('columns'):
        columns = session['columns']

    return (render_template('search.htm', 
                            title= 'Keyword Finder',
                            tables=tables, 
                            columns=columns, 
                            new_table=new_table, 
                            shape_rows= shape_rows, 
                            shape_cols= shape_cols,
                            filename=filename))

@app.route('/remove_duplicate', methods=['GET','POST'])
def duplicate():
    # Initialize variables for template rendering
    table_duplicate = new_table_duplicate = columns_duplicate= shape_rows = shape_cols= filename = num_duplicates=None
    if request.method == 'POST' and 'upload' in request.form:
        # Clear old session values if they exist
        session_list= ['selected_columns', 'selected_choice', 'modified_df']    
        clear_session(session_list)
        table_duplicate, columns_duplicate, shape, filename = handle_file_upload(request, UPLOAD_FOLDER)
        shape_rows= shape[0]
        shape_cols = shape[1]
    elif request.method == 'POST' and 'process' in request.form:
            #num_duplicates = df.duplicated(subset=['col1', 'col2']).sum()
        selected_columns = request.form.get('column')
        selected_choice = request.form.get('row')
        if selected_choice == 'false':
            selected_choice = False
        # Save all selections in session
        session['selected_columns'] = selected_columns
        session['selected_choice'] = selected_choice
        filename = session.get('uploaded_file_name')        
        filepath = session.get('data_file')
        if filepath and selected_columns:
            df_a = pd.read_excel(filepath)
            num_duplicates = df_a.duplicated(subset=selected_columns).sum()
            new_df = df_a.drop_duplicates(subset= selected_columns, keep=selected_choice)
            new_table_duplicate = new_df.to_html(classes='table table-bordered table-striped table-hover align-middle', index=False, border=0)
            session['modified_df'] = new_df.to_csv(index=False)
            shape = new_df.shape
            shape_rows = shape[0]
            shape_cols = shape[1]
        # Retrieve columns from session if not set (for dropdown persistence)
    if columns_duplicate is None and session.get('columns'):
        columns_duplicate = session['columns']

    return (render_template('duplicate.htm',
                            title= 'Duplicate Cleaner',
                            table_duplicate=table_duplicate, 
                            columns_duplicate=columns_duplicate,
                            shape_rows = shape_rows,
                            shape_cols = shape_cols, 
                            new_table_duplicate=new_table_duplicate,
                            filename=filename,
                            num_duplicates=num_duplicates))


@app.route('/dual', methods=['POST', 'GET'])
def dual_keywords():
        # Initialize variables for template rendering
    table = new_table = columns= shape_rows = shape_cols= filename = None
    if request.method == 'POST' and 'upload' in request.form:
        # Clear old session values if they exist
        session_list= ['selected_columns', 'keywords1', 'keywords2', 'modified_df']    
        clear_session(session_list)
        table, columns, shape, filename = handle_file_upload(request, UPLOAD_FOLDER)
        shape_rows= shape[0]
        shape_cols = shape[1]

    elif request.method == 'POST' and 'process' in request.form:
        selected_column = request.form.get('column')
        keywords1 = [k.strip() for k in request.form.get('keywords1', '').split(',')]
        keywords2 = [k.strip() for k in request.form.get('keywords2', '').split(',')]
        # Save all selections in session
        session['selected_columns'] = selected_column
        session['keywords1'] = keywords1
        session['keywords2'] = keywords2
        filepath = session.get('data_file')
        filename = session.get('uploaded_file_name')        
        if filepath and selected_column:
            df_a = pd.read_excel(filepath)
            new_df = filter_rows_by_dual_keywords(df_a, keywords1, keywords2, selected_column)
            new_table = new_df.to_html(classes='table table-bordered table-striped table-hover align-middle', index=False, border=0)
            session['modified_df'] = new_df.to_csv(index=False)
            shape = new_df.shape
            shape_rows = shape[0]
            shape_cols = shape[1]
        # Retrieve columns from session if not set (for dropdown persistence)
    if columns is None and session.get('columns'):
        columns = session['columns']

    return (render_template('dual.htm',
                            title= 'Co-Occurrence Detector',
                            table=table, 
                            columns=columns,
                            shape_rows = shape_rows,
                            shape_cols = shape_cols, 
                            new_table=new_table,
                            filename=filename))

@app.route('/article_frequency_analyzer', methods=['GET', "POST"])
def article_frequency_analyzer():
    # Initialize variables for template rendering
    table = new_table = columns= shape_rows = shape_cols= filename= start_date= end_date= plot_path=download_btn= None
    freq = 'M'
    latest_date = earliest_date = None
    if request.method == 'POST' and 'upload' in request.form:
        # Clear old session values if they exist
        session_list= ['selected_column', 'start_date', 'end_date', 'modified_df']    
        clear_session(session_list)
        table, columns, shape, filename = handle_file_upload(request, UPLOAD_FOLDER)
        shape_rows= shape[0]
        shape_cols = shape[1]
    elif request.method == 'POST' and 'process' in request.form:
        selected_column = request.form.get('column')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        freq= request.form.get('freq')
        # Save all selections in session
        session['selected_column'] = selected_column
        session['freq']= freq
        session['start_date'] = start_date
        session['end_date'] = end_date
        filepath = session.get('data_file')
        filename = session.get('uploaded_file_name')        
        if filepath and selected_column:
            df_a = pd.read_excel(filepath)
            new_df, latest_date, earliest_date = articles_per_period(df_a, selected_column, freq=freq, start_date=start_date, end_date=end_date)
            new_table = new_df.to_html(classes='table table-bordered table-striped table-hover align-middle', index=False, border=0)
            download_btn = True
            session['modified_df'] = new_df.to_csv(index=False)

            # --- Create the visualization ---
            plt.figure(figsize=(10, 5))
            plt.bar(new_df.columns, new_df.iloc[0], color='#007bff')
            plt.title('Article Frequency Over Time')
            plt.xlabel('Period')
            plt.ylabel('Number of Articles')
            plt.xticks(rotation=90, ha='right')
            plt.tight_layout()

            # Save plot to static folder
            plot_filename = 'frequency_plot.png'
            plot_path = os.path.join(app.root_path, 'static', plot_filename)
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()

            plot_path = url_for('static', filename=plot_filename)

        # Retrieve columns from session if not set (for dropdown persistence)
    if columns is None and session.get('columns'):
        columns = session['columns']
    return (render_template('frequency.htm',
                            title= 'Publication Trend Analyzer',
                            table=table, 
                            columns=columns,
                            shape_rows = shape_rows,
                            shape_cols = shape_cols, 
                            new_table=new_table,
                            filename=filename,
                            plot_path=plot_path,
                            download_btn=download_btn,
                            latest_date= latest_date,
                            earliest_date= earliest_date)) 

@app.route('/data-overview', methods=['POST'])
def data_overview():
    filename = session.get('uploaded_file_name')
    csv_data = session.get('modified_df')
    if not csv_data:
        return render_template('data-overview.htm', error="No data available.")

    # Read CSV from session
    df = pd.read_csv(io.StringIO(csv_data))

    # Basic info
    num_rows, num_cols = df.shape
    #Start
    #check why data-overview not working for search and duplicated pages
    #to specify columns to consider duplicated
    #num_duplicates = df.duplicated(subset=['col1', 'col2']).sum()
    num_duplicates = df.duplicated().sum()
    missing_counts = df.isnull().sum().to_dict()
    dtypes = df.dtypes.to_dict()
    unique_counts = df.nunique().to_dict()

    # Top frequent values per column (first 5)
    top_values = {}
    for col in df.columns:
        top_values[col] = df[col].value_counts().head(5).to_dict()

    # Generate missingno visualization and save as static image
    plot_path = os.path.join(app.root_path, 'static', 'missingno_plot.png')
    msno.matrix(df)
    plt.savefig(plot_path)
    plt.close()

    return render_template('data-overview.htm',
                           shape_rows=num_rows,
                           shape_cols=num_cols,
                           num_duplicates=num_duplicates,
                           missing_counts=missing_counts,
                           dtypes=dtypes,
                           unique_counts=unique_counts,
                           top_values=top_values,
                           plot_path=plot_path,
                           title='Data Overview',
                           filename= filename)


@app.route('/download', methods=['POST'])
def download():
    # Assume the modified DataFrame is stored in session as CSV string
    csv_data = session.get('modified_df')
    if csv_data:
        # Read CSV string into DataFrame
        df = pd.read_csv(io.StringIO(csv_data))

        # Create in-memory Excel file
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        output.seek(0)

        # Send the file to the user
        return send_file(output,
                         download_name="modified_data.xlsx",
                         as_attachment=True,
                         mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    return "No data available", 400
