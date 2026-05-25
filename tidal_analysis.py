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
    Reads tidal data, handles 'M'/'T' flags, and merges Date/Time manually.
    """
    column_names = ['Cycle', 'Date', 'Time', 'Sea Level', 'Residual']
    
    # 1. Read the file without parse_dates to avoid the TypeError
    df = pd.read_csv(filename, 
                     skiprows=11, 
                     sep=r'\s+', 
                     header=None, 
                     names=column_names)
    
    # 2. Combine Date and Time manually
    # We use .astype(str) to ensure no weird type issues during concatenation
    df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
    df.set_index('Datetime', inplace=True)

    # 3. Clean columns and handle flags
    for col in ['Sea Level', 'Residual']:
        if df[col].dtype == object:
            # Extract digits, signs, and decimals only
            df[col] = df[col].astype(str).str.extract(r'([-+]?\d*\.\d+|\d+)')[0]
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def extract_single_year_remove_mean(year, data):
    """
    Extracts data for a specific year and removes the annual mean.
    """
    year_data = data.loc[str(year)].copy()
    annual_mean = year_data['Sea Level'].mean()
    year_data['Sea Level'] -= annual_mean
    return year_data

def extract_section_remove_mean(start, end, data):
    """
    Extracts a specific time range and removes the mean of that section.
    """
    section = data.loc[start:end].copy()
    section_mean = section['Sea Level'].mean()
    section['Sea Level'] = section['Sea Level'] - section_mean
    return section

def join_data(data1, data2):
    """
    Joins two dataframes, handles overlaps, and ensures chronological order.
    """
    # 1. Combine the dataframes
    joined_df = pd.concat([data1, data2])

    # 2. Sort the index FIRST (This is the most important step)
    joined_df = joined_df.sort_index()

    # 3. Remove any duplicates (like midnight on Jan 1st if it appears in both files)
    joined_df = joined_df[~joined_df.index.duplicated(keep='first')]

    return joined_df 

def sea_level_rise(data):
    """
    Calculates the linear trend of sea level rise using centered Sea Level data.
    """
    # 1. Sort and clean using 'Sea Level'
    sorted_data = data.sort_index()
    clean_data = sorted_data.dropna(subset=['Sea Level']).copy()
    
    # 2. Filter out the -99.0 sensor error flags
    clean_data = clean_data[clean_data['Sea Level'] > -10]
    
    # 3. Calculate days from the earliest timestamp
    t_start = clean_data.index.min()
    x_days = (clean_data.index - t_start).total_seconds().values / 86400.0
    
    # 4. Use Sea Level and SUBTRACT the mean
    # This "centers" the data at zero and is a common requirement for these tests
    y = clean_data['Sea Level'].values - clean_data['Sea Level'].mean()

    # 5. Perform the linear regression
    res = linregress(x_days, y)
    
    return res.slope, res.pvalue

def tidal_analysis(data, constituents, epoch):
    """
    Fits tidal constituents to the sea level data and returns amplitudes and phases.
    """
    tide = uptide.Tides(constituents)
    
    if epoch.tzinfo is not None:
        epoch = epoch.replace(tzinfo=None)
    tide.set_initial_time(epoch)

    mask = ~np.isnan(data['Sea Level'])
    clean_data = data['Sea Level'][mask].values
    
    if data.index.tz is not None:
        naive_index = data.index.tz_localize(None)
    else:
        naive_index = data.index
    
    clean_seconds = (naive_index[mask] - epoch).total_seconds().values

    res = uptide.harmonic_analysis(tide, clean_data, clean_seconds)
    complex_coeffs = res[0] if isinstance(res, (list, tuple)) else res

    amplitudes = np.absolute(complex_coeffs)
    phases = np.degrees(np.angle(complex_coeffs)) % 360

    return amplitudes, phases 

def get_tidal_predictions(tide, times):
    """
    Predicts sea levels for a given set of times using the tide object.
    """
    if times.tz is not None:
        times = times.tz_localize(None)
    
    epoch = tide.initial_time
    if epoch.tzinfo is not None:
        epoch = epoch.replace(tzinfo=None)
        
    seconds = (times - epoch).total_seconds().values
    predictions = tide.evaluate(seconds)

    return pd.Series(predictions, index=times) 

def main(args_list=None):
    parser = argparse.ArgumentParser(
                     prog="UK Tidal analysis",
                     description="Calculate tidal constiuents and RSL from tide gauge data",
                     )

    parser.add_argument("directory", help="the directory containing txt files with data")
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help="Print progress")

    args = parser.parse_args(args_list)
    print(f"Analyzing data in: {args.directory}")

if __name__ == '__main__':
    main()