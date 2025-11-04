from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import os
from werkzeug.utils import secure_filename
import re
from flask import send_file
import io
import numpy as np

def clear_session(list):
    for key in list:
        session.pop(key, None)

def handle_file_upload(request, upload_folder):
    """Handle Excel file upload and return (tables, columns) for preview."""
    clear_upload_folder(upload_folder)
    file = request.files.get('file')

    if not file or not file.filename.endswith('.xlsx'):
        return None, None  # Invalid file

    filename = secure_filename(file.filename)
    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)

    # Save info in session
    session['data_file'] = filepath
    session['uploaded_file_name'] = filename

    # Generate preview table
    df = pd.read_excel(filepath)
    tables = df.to_html(
        classes='table table-bordered table-striped table-hover align-middle',
        index=False,
        border=0
    )
    #Shape
    shape = df.shape
    # Get column names
    columns = pd.read_excel(filepath, nrows=0).columns.tolist()
    session['columns'] = columns

    return tables, columns, shape, filename

def clear_upload_folder(upload_folder):
    """
    Deletes all files in the given upload folder.
    """
    for filename in os.listdir(upload_folder):
        file_path = os.path.join(upload_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # remove file or symlink
            elif os.path.isdir(file_path):
                # Optional: remove directories if needed
                import shutil
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def new_column_match_keywords(df, column_name, new_column, keywords,
                              no_match_keyword=None, regex='search',
                              case_sensitive=False):
    
    df[column_name] = df[column_name].astype(str)
    pattern = rf"\b({'|'.join(keywords)})\b"
    flags = 0 if case_sensitive else re.IGNORECASE

    if regex == 'search':
        df[new_column] = df[column_name].str.extract(pattern, flags=flags)[0]
    elif regex == 'findall':
        df[new_column] = df[column_name].str.findall(pattern, flags=flags)
        df = df.explode(new_column).reset_index(drop=True)
    else:
        raise ValueError("regex must be either 'search' or 'findall'")

    if no_match_keyword is not None:
        df[new_column] = df[new_column].fillna(no_match_keyword)

    return df


def filter_rows_by_dual_keywords(df, keywords1, keywords2, column_name):

    """
    Keep only rows where 'column_name' contains at least one keyword 
    from both keywords1 and keywords2 lists.
    """
    pattern1 = '|'.join(map(re.escape, keywords1))
    pattern2 = '|'.join(map(re.escape, keywords2))
    
    mask1 = df[column_name].str.contains(pattern1, flags=re.IGNORECASE, na=False, regex=True)
    mask2 = df[column_name].str.contains(pattern2, flags=re.IGNORECASE, na=False, regex=True)
    
    return df[mask1 & mask2].reset_index(drop=True)

def articles_per_period(df: pd.DataFrame, 
                        date_column: str='Published Date', 
                        freq: str='M', 
                        start_date=None, 
                        end_date=None) -> pd.DataFrame:
    """
    Counts number of articles per given period in the DataFrame.

    Parameters:
    - df: pd.DataFrame with date column
    - date_column: str, name of the date column
    - freq: str, frequency for resampling: 'D' (day), 'M' (month), 'Y' (year)
    - start_date, end_date: optional, string or datetime

    Returns:
    - pd.DataFrame: single-row with period labels as columns
    """
    # Ensure datetime
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df = df.dropna(subset=[date_column])
    
    # Set index
    df = df.set_index(date_column)
    
    # Resample
    articles_count = df.resample(freq).size()
    earliest_date = articles_count.index.min()
    latest_date = articles_count.index.max()
    # Handle start/end dates
    if not start_date:
        start_date = articles_count.index.min()
    else:
        start_date = pd.to_datetime(start_date, errors='coerce')

    if not end_date:
        end_date = articles_count.index.max()
    else:
        end_date = pd.to_datetime(end_date, errors='coerce')
    
    # Create full date range
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    articles_count = articles_count.reindex(date_range, fill_value=0)
    
    # Format column labels
    if freq == 'D':
        col_labels = articles_count.index.strftime('%Y-%m-%d')
    elif freq == 'M':
        col_labels = articles_count.index.strftime('%Y-%m')
    elif freq == 'Y':
        col_labels = articles_count.index.strftime('%Y')
    
    return pd.DataFrame([articles_count.values], columns=col_labels), latest_date, earliest_date

#Extract keywords functions

def combined_text(df, text_columns):
  df['combined_text']= df[text_columns].astype(str).agg(' '.join)
  df['combined_text'] = df['combined_text'].str.strip()
  df_result = df.drop(columns=text_columns)
  column_name = 'combined_text'
  return df_result, column_name

def find_keywords(df, column_name, keywords):
  df[column_name] = df[column_name].astype(str)
  pattern = rf"\b({'|'.join(keywords)})\b"
  df['keywords'] = df[column_name].str.findall(pattern, flags=re.IGNORECASE)
  df= df[df['keywords'].str.len()>0]
  return df

def expload_sentences(df, column_name):
    def sentences(text):
        text = text.replace('.', f".\n")
        return [s.strip() for s in text.split('\n') if s.strip()]
    df['sentences']= df[column_name].apply(sentences)
    df= df.explode('sentences')
    df = df.explode('keywords')
    df = df[df['sentences'].str.len() > 0]
    df = df.drop_duplicates()
    return df

def occurences(df):
    mask = df.apply(
        lambda row: row['keywords'].lower() in row['sentences'].lower(),
        axis=1
    )
    df = df[mask]

    # Only proceed if df is not empty
    if not df.empty:
        df['occurrences'] = np.vectorize(
            lambda s, k: s.lower().count(k.lower()),
            otypes=[int]
        )(df['sentences'], df['keywords'])
    else:
        df['occurrences'] = []  # keep structure consistent
    return df

def extract_keywords(df, text_columns, keywords):
  #combined_text_df, column_name = combined_text(df, text_columns)
  column_name='Body 1'
  combined_text_df= df
  keywords_df = find_keywords(combined_text_df, column_name, keywords)
  exploaded_df = expload_sentences(keywords_df, column_name)
  #occurence_keywords_df= occurences(exploaded_df)
  occurence_keywords_df = exploaded_df.drop(column_name, axis=1)
  col1, col2 = 'sentences', 'keywords'
  cols = list(occurence_keywords_df.columns)
  i1, i2 = cols.index(col1), cols.index(col2)
  cols[i1], cols[i2] = cols[i2], cols[i1]  # swap positions
  df = occurence_keywords_df[cols]
  return df







