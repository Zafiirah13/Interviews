import os
import pandas as pd


def list_directory(path):
    '''
    INPUT:
        path (str): The directory where the file is located
    OUTPUT:
        list_dir (list): List of files that contains in the specific directory
    '''
    
    list_dir = os.listdir(path)
    
    return list_dir

def combine_metadata(path):
    
    '''
    Input: 
        path (str): The directory where the metadata file is
    
    Output:
        combined_data (df): A table containing sorted combine metadata by claim ID
    '''
    
    # List the number of file in meta_data folder
    metadata_file = list_directory(path)
    
    # Empty dataframe
    combined_data = pd.DataFrame()
        
    # Read each file and append in an empty data frame
    for i in range(len(metadata_file)):
        file = pd.read_csv(path+metadata_file[i])
        combined_data = combined_data.append(file, ignore_index=True)
        
    # Check the shape of the file
    print('The combined metadata has shape: ',combined_data.shape)
    
    # Sort Claim ID to check for duplicates. This mean same Claim and Car
    combined_data = combined_data.sort_values('claim_id')
    return combined_data

def read_csv(path, filename):
    '''
    Input:
        path     (str) : The directory of the csv file
        filename (str) : The name of the csv file
        
    Output:
        file (df): A dataframe of the csv file    
    '''
    
    file = pd.read_csv(path+filename)
    print('The file has shape: ',file.shape)
    return file

