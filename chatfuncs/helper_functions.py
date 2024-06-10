import os
import re
import pandas as pd
import gradio as gr

import os
import shutil

import os
import shutil
import getpass
import gzip
import pickle

def get_or_create_env_var(var_name, default_value):
    # Get the environment variable if it exists
    value = os.environ.get(var_name)
    
    # If it doesn't exist, set it to the default value
    if value is None:
        os.environ[var_name] = default_value
        value = default_value
    
    return value

# Retrieving or setting output folder
env_var_name = 'GRADIO_OUTPUT_FOLDER'
default_value = 'output/'

output_folder = get_or_create_env_var(env_var_name, default_value)
print(f'The value of {env_var_name} is {output_folder}')

def ensure_output_folder_exists(output_folder):
    """Checks if the output folder exists, creates it if not."""

    folder_name = output_folder

    if not os.path.exists(folder_name):
        # Create the folder if it doesn't exist
        os.makedirs(folder_name)
        print(f"Created the output folder:", folder_name)
    else:
        print(f"The output folder already exists:", folder_name)

# Attempt to delete content of gradio temp folder
def get_temp_folder_path():
    username = getpass.getuser()
    return os.path.join('C:\\Users', username, 'AppData\\Local\\Temp\\gradio')

def empty_folder(directory_path):
    if not os.path.exists(directory_path):
        #print(f"The directory {directory_path} does not exist. No temporary files from previous app use found to delete.")
        return

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            #print(f'Failed to delete {file_path}. Reason: {e}')
            print('')



def get_file_path_end(file_path):
    # First, get the basename of the file (e.g., "example.txt" from "/path/to/example.txt")
    basename = os.path.basename(file_path)
    
    # Then, split the basename and its extension and return only the basename without the extension
    filename_without_extension, _ = os.path.splitext(basename)

    #print(filename_without_extension)
    
    return filename_without_extension

def get_file_path_end_with_ext(file_path):
    match = re.search(r'(.*[\/\\])?(.+)$', file_path)
        
    filename_end = match.group(2) if match else ''

    return filename_end

def detect_file_type(filename):
    """Detect the file type based on its extension."""
    if (filename.endswith('.csv')) | (filename.endswith('.csv.gz')) | (filename.endswith('.zip')):
        return 'csv'
    elif filename.endswith('.xlsx'):
        return 'xlsx'
    elif filename.endswith('.parquet'):
        return 'parquet'
    elif filename.endswith('.pkl.gz'):
        return 'pkl.gz'
    else:
        raise ValueError("Unsupported file type.")

def read_file(filename):
    """Read the file based on its detected type."""
    file_type = detect_file_type(filename)
        
    print("Loading in file")

    if file_type == 'csv':
        file = pd.read_csv(filename, low_memory=False).reset_index(drop=True).drop(["index", "Unnamed: 0"], axis=1, errors="ignore")
    elif file_type == 'xlsx':
        file = pd.read_excel(filename).reset_index(drop=True).drop(["index", "Unnamed: 0"], axis=1, errors="ignore")
    elif file_type == 'parquet':
        file = pd.read_parquet(filename).reset_index(drop=True).drop(["index", "Unnamed: 0"], axis=1, errors="ignore")
    elif file_type == 'pkl.gz':
        with gzip.open(filename, 'rb') as file:
            file = pickle.load(file)
            #file = pd.read_pickle(filename)

    print("File load complete")

    return file

def put_columns_in_df(in_file, in_bm25_column):
    '''
    When file is loaded, update the column dropdown choices and change 'clean data' dropdown option to 'no'.
    '''

    file_list = [string.name for string in in_file]

    #print(file_list)

    data_file_names = [string for string in file_list]
    data_file_name = data_file_names[0]

    new_choices = []
    concat_choices = []
    
    
    df = read_file(data_file_name)

    new_choices = list(df.columns)


    concat_choices.extend(new_choices)     
        
    return gr.Dropdown(choices=concat_choices), df

def put_columns_in_join_df(in_file, in_bm25_column):
    '''
    When file is loaded, update the column dropdown choices and change 'clean data' dropdown option to 'no'.
    '''

    print("in_bm25_column")

    new_choices = []
    concat_choices = []
    
    
    df = read_file(in_file.name)
    new_choices = list(df.columns)

    print(new_choices)

    concat_choices.extend(new_choices)     
        
    return gr.Dropdown(choices=concat_choices)

def dummy_function(gradio_component):
    """
    A dummy function that exists just so that dropdown updates work correctly.
    """
    return None    

def display_info(info_component):
    gr.Info(info_component)

