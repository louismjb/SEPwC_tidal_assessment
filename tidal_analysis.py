# import the modules we need
import pandas as pd
import datetime
import os
import numpy as np
import uptide
import pytz
import math
from scipy import stats
import matplotlib.dates as mdates
import argparse


def read_tidal_data(filename):
    """
    Reads tidal data from a file.
    """
    # 1. Read the file
    # We use r'\s+' for the separator and names=... to label the columns
    column_names = ['Cycle', 'Date', 'Time', 'Sea Level', 'Residual']
    
    df = pd.read_csv(filename, 
                     skiprows=11, 
                     sep=r'\s+', 
                     header=None, 
                     names=column_names)
    
    # Combine Date and Time columns into one
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    
    # Set the Datetime column as the index
    df.set_index('Datetime', inplace=True)
    
    # Replace values ending in M, T, or N with NaN (Not a Number)
    df.replace(to_replace="[a-zA-Z]$", value={'Sea Level': np.nan}, regex=True, inplace=True)
    
    # Ensure the Sea Level column is treated as numbers (floats)
    df['Sea Level'] = pd.to_numeric(df['Sea Level'], errors='coerce')
    
    return df
    
def extract_single_year_remove_mean(year, data):
    """
    Extracts data for a specific year and removes the annual mean.

    Args:
        year (int or str): The year to extract.
        data (pd.DataFrame): The input dataframe with a DatetimeIndex.

    Returns:
        pd.DataFrame: A copy of the data for the specified year with the mean removed.
    """
    # Select only the rows for the given year and create a deep copy
    year_data = data.loc[str(year)].copy()

    # Calculate the mean sea level, automatically skipping any NaN values
    annual_mean = year_data['Sea Level'].mean()

    # Subtract the mean from the 'Sea Level' column (Broadcasting)
    year_data['Sea Level'] = year_data['Sea Level'] - annual_mean

    return year_data

def extract_section_remove_mean(start, end, data):

    return year_data


def join_data(data1, data2):
    """
    Joins two dataframes together and sorts by the index.
    """
    # 1. Stack data1 and data2 on top of each other
    joined_df = pd.concat([data1, data2])

    # 2. Sort the index (so the years are in chronological order)
    joined_df.sort_index(inplace=True)

    # 3. Remove any duplicate rows that might exist
    joined_df = joined_df[~joined_df.index.duplicated(keep='first')]

    return joined_df

def sea_level_rise(data):

    return

def tidal_analysis(data, constituents, start_datetime):

    return

def get_longest_contiguous_data(data):

    return 


def main(args_list=None):

    parser = argparse.ArgumentParser(
                     prog="UK Tidal analysis",
                     description="Calculate tidal constiuents and RSL from tide gauge data",
                     )

    parser.add_argument("directory",
                    help="the directory containing txt files with data")
    parser.add_argument('-v', '--verbose',
                    action='store_true',
                    default=False,
                    help="Print progress")

    args = parser.parse_args(args_list)
    dirname = args.directory
    verbose = args.verbose

    print("Add your code here to do things!")
    

if __name__ == '__main__':
    main()
