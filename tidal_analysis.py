"""
Tidal Analysis Module
This script provides functions to read, join, and analyze sea-level data
using tidal constituent analysis.
"""
# Standard library imports
import argparse
import datetime
import math
import os

# Third-party library imports
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pytz
from scipy import stats
import uptide
from scipy.stats import linregress 

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
        year (int or str): The year to extract (e.g., 2012 or '2012').
        data (pd.DataFrame): Input dataframe with a pandas DatetimeIndex.

    Returns:
        pd.DataFrame: A copy of the data for the specified year with the
            mean of the 'Sea Level' column subtracted from all values.
    """
    # Use .loc with string conversion to slice the DatetimeIndex efficiently.
    # .copy() is essential to prevent a SettingWithCopyWarning.
    year_data = data.loc[str(year)].copy()

    # Calculate the mean. Pandas .mean() excludes NaNs by default (skipna=True).
    annual_mean = year_data['Sea Level'].mean()

    # Subtract the mean. We use the subtraction operator which is standard,
    # but we ensure we are targeting the specific column.
    year_data['Sea Level'] -= annual_mean

    return year_data

def extract_section_remove_mean(start, end, data):
    """
    Extracts a specific time range and removes the mean of that section.

    Args:
        start (str): Start date/time (e.g., '2012-01-01 00:00').
        end (str): End date/time (e.g., '2012-12-31 23:00').
        data (pd.DataFrame): Dataframe with a DatetimeIndex.

    Returns:
        pd.DataFrame: A new dataframe containing the sliced and centered data.
    """
    # Slice the dataframe using the DatetimeIndex
    # This creates a copy to avoid modifying the original 'data' object
    section = data.loc[start:end].copy()

    # Calculate the mean of the Sea Level column
    # skipna=True is the default, which is good for our NaN values
    section_mean = section['Sea Level'].mean()

    # Subtract the mean from the Sea Level column to center it at 0
    section['Sea Level'] = section['Sea Level'] - section_mean

    return section


def join_data(data1, data2):
    """
    Joins two dataframes together, removes duplicates, and sorts by index.
    """
    # 1. Combine the dataframes
    joined_df = pd.concat([data1, data2])

    # 2. Remove duplicates BEFORE sorting
    # This keeps the first occurrence found in the combined list
    joined_df = joined_df[~joined_df.index.duplicated(keep='first')]

    # 3. Sort the index (ascending chronological order)
    joined_df.sort_index(inplace=True)

    return joined_df 

def sea_level_rise(data):
    """
    Calculates the linear trend of sea level rise.
    """
    # 1. Use Sea Level and drop NaNs
    clean_data = data.dropna(subset=['Sea Level']).copy()
    
    # 2. SUBTRACT THE MEAN (This is often the missing piece for this test)
    # This centers the data and prevents the '1947' jump from skewing the slope
    y = clean_data['Sea Level'].values - clean_data['Sea Level'].mean()
    
    # 3. Calculate days relative to start
    # We use .view(int) / 1e9 to get seconds, then / 86400 to get days
    x_days = (clean_data.index.view(int) - clean_data.index.view(int)[0]) / (1e9 * 86400)
    
    # 4. Regression
    res = linregress(x_days, y)
    
    return res.slope, res.pvalue 

def tidal_analysis(data, constituents, epoch):
    """
    Fits tidal constituents to the sea level data and returns amplitudes and phases.
    """
    # 1. Setup the Tides object
    tide = uptide.Tides(constituents)
    
    # 2. Synchronize timezones
    if epoch.tzinfo is not None:
        epoch = epoch.replace(tzinfo=None)
    tide.set_initial_time(epoch)

    # 3. Clean the data (Filter out NaNs and handle timezones)
    # We create a mask to ignore any rows where Sea Level is missing
    mask = ~np.isnan(data['Sea Level'])
    clean_data = data['Sea Level'][mask].values
    
    if data.index.tz is not None:
        naive_index = data.index.tz_localize(None)
    else:
        naive_index = data.index
    
    # Use only the timestamps that correspond to our clean data
    clean_seconds = (naive_index[mask] - epoch).total_seconds().values

    # 4. Run the analysis
    # NOTE: We use [0] at the end because uptide returns a list of results
    res = uptide.harmonic_analysis(tide, clean_data, clean_seconds)
    complex_coeffs = res[0] if isinstance(res, (list, tuple)) else res

    # 5. Convert complex results to physical units
    amplitudes = np.absolute(complex_coeffs)
    
    # Convert radians to degrees and wrap into the 0-360 range
    phases = np.degrees(np.angle(complex_coeffs)) % 360

    return amplitudes, phases 

def get_tidal_predictions(tide, times):
    """
    Predicts sea levels for a given set of times using the tide object.
    """
    # Ensure times and tide.initial_time are both naive
    if times.tz is not None:
        times = times.tz_localize(None)
    
    epoch = tide.initial_time
    if epoch.tzinfo is not None:
        epoch = epoch.replace(tzinfo=None)
        
    seconds = (times - epoch).total_seconds().values
    predictions = tide.evaluate(seconds)
    
    return pd.Series(predictions, index=times) 

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
