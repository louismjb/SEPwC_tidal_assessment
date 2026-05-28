"""
Tidal Analysis Tool
Author: Louis Johnson Brickhill
Description: This script performs harmonic analysis and sea-level rise 
             calculations on BODC format tidal data files.
Usage: python3 tidal_analysis.py [-v] <directory_path>
"""
# Standard library imports
import argparse
# Kept to prevent a NameError in test_tides.py due to a missing test dependency
import datetime  # pylint: disable=unused-import
import os
import sys

# Third-party imports
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import uptide
from scipy.stats import linregress

def read_tidal_data(file_path):
    """
    Reads tidal elevation data from a BODC format text file.

    Parses the date and time into a DatetimeIndex and handles numeric 
    conversion for sea-level records. Keeps original columns to maintain 
    compatibility with existing test drop requirements.

    Args:
        file_path (str): Path to the tidal data .txt file.

    Returns:
        pd.DataFrame: A DataFrame containing 'Date', 'Time', and 'Sea Level' 
            columns, indexed by a 'datetime' DatetimeIndex.
    """
    data = pd.read_csv(
        file_path,
        sep=r'\s+',
        skiprows=11,
        names=['RowID', 'Date', 'Time', 'Sea Level', 'Residual'],
        usecols=['Date', 'Time', 'Sea Level']
    )

    data['Sea Level'] = pd.to_numeric(data['Sea Level'], errors='coerce')
    data['datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'],
                                     errors='coerce')
    data.set_index('datetime', inplace=True)

    return data[['Date', 'Time', 'Sea Level']]

def extract_single_year_remove_mean(year, data):
    """
    Extracts data for a specific year and centers it by removing the mean.

    Args:
        year (int): The calendar year to extract.
        data (pd.DataFrame): The full dataset indexed by datetime.

    Returns:
        pd.DataFrame: A DataFrame reindexed to a full hourly range for the year 
            with the annual mean subtracted from 'Sea Level'.
    """
    start_date = f"{year}-01-01 00:00:00"
    end_date = f"{year}-12-31 23:00:00"
    full_year_range = pd.date_range(start=start_date, end=end_date, freq='h')

    year_data = data.reindex(full_year_range)
    annual_mean = year_data['Sea Level'].mean()
    year_data['Sea Level'] -= annual_mean
    year_data.index.name = 'datetime'

    return year_data

def extract_section_remove_mean(start, end, data):
    """
    Extracts a custom date range and centers the sea-level values.

    Args:
        start (str): Start date string (e.g., '2018-01-01').
        end (str): End date string.
        data (pd.DataFrame): The full dataset.

    Returns:
        pd.DataFrame: The extracted section with the mean removed.
    """
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    if end_dt.hour == 0 and end_dt.minute == 0:
        end_dt = end_dt + pd.Timedelta(hours=23)

    full_range = pd.date_range(start=start_dt, end=end_dt, freq='h')
    section = data.reindex(full_range)
    section_mean = section['Sea Level'].mean()
    section['Sea Level'] = section['Sea Level'] - section_mean
    section.index.name = 'datetime'

    return section

def join_data(data1, data2):
    """
    Merges two datasets into a single, contiguous hourly timeline.

    Args:
        data1 (pd.DataFrame): First tidal dataset.
        data2 (pd.DataFrame): Second tidal dataset.

    Returns:
        pd.DataFrame: A merged, sorted DataFrame reindexed to include every 
            hour between the absolute start and end times.
    """
    joined_df = pd.concat([data1, data2])
    joined_df = joined_df.sort_index()
    joined_df = joined_df[~joined_df.index.duplicated(keep='first')]

    full_range = pd.date_range(start=joined_df.index.min(),
                               end=joined_df.index.max(),
                               freq='h')

    joined_df = joined_df.reindex(full_range)
    joined_df.index.name = 'datetime'

    return joined_df

def sea_level_rise(data):
    """
    Performs linear regression to find the sea-level trend over time.

    Args:
        data (pd.DataFrame): Dataset with 'Sea Level' and DatetimeIndex.

    Returns:
        tuple: (slope, p_value) where slope is meters per day.
    """
    clean_data = data.dropna(subset=['Sea Level'])

    if len(clean_data) < 2:
        return 0.0, 1.0

    days = mdates.date2num(clean_data.index.to_pydatetime())
    slope, _, _, p_value, _ = linregress(days, clean_data['Sea Level'])

    return slope, p_value

def tidal_analysis(data, constituents, epoch):
    """
    Conducts harmonic analysis for specified tidal constituents.

    Args:
        data (pd.DataFrame): Cleaned sea-level data.
        constituents (list): List of constituent names (e.g., ['M2', 'S2']).
        epoch (datetime.datetime): The initial time reference for the analysis.

    Returns:
        tuple: (amplitudes, phases) as numpy arrays.
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
    Generates predicted sea levels for specific timestamps.

    Args:
        tide (uptide.Tides): A fitted uptide Tides object.
        times (pd.DatetimeIndex): The timestamps for which to predict levels.

    Returns:
        pd.Series: Predicted sea levels indexed by time.
    """
    if times.tz is not None:
        times = times.tz_localize(None)

    epoch = tide.initial_time
    if epoch.tzinfo is not None:
        epoch = epoch.replace(tzinfo=None)

    seconds = (times - epoch).total_seconds().values
    predictions = tide.evaluate(seconds)

    return pd.Series(predictions, index=times)

def calculate_tidal_components(data):
    """
    Simplifies harmonic analysis specifically for M2 and S2 components.

    Args:
        data (pd.DataFrame): Tidal dataset.

    Returns:
        tuple: (m2_amplitude, s2_amplitude) in meters.
    """
    clean_data = data.dropna(subset=['Sea Level'])

    if clean_data.empty or len(clean_data) < 100:
        return 0.0, 0.0

    levels = clean_data['Sea Level'].values.astype(float)
    t0 = clean_data.index[0]
    t_seconds = (clean_data.index - t0).total_seconds().values.astype(float)

    tide = uptide.Tides(['M2', 'S2'])
    tide.set_initial_time(t0.to_pydatetime())

    try:
        coeffs = uptide.harmonic_analysis(tide, levels, t_seconds)
        if isinstance(coeffs, tuple):
            coeffs = coeffs[0]

        return float(np.abs(coeffs[0])), float(np.abs(coeffs[1]))

    except Exception as error:
        sys.stderr.write(f"Harmonic analysis failed: {error}\n")
        raise

def find_longest_contiguous_period(data):
    """
    Determines the largest number of consecutive hourly records.

    Args:
        data (pd.DataFrame): Dataset indexed by datetime.

    Returns:
        int: Count of records in the longest uninterrupted stretch.
    """
    if data.empty:
        return 0
    time_diffs = data.index.to_series().diff()
    is_break = time_diffs > time_diffs.median()
    group_ids = is_break.cumsum()
    return data.groupby(group_ids).size().max()

def main(args_list=None):
    """
    Coordinates data loading, analysis, and reporting for a tidal station.

    Scans a directory for BODC data files, performs harmonic analysis and 
    linear regression, and outputs a quality-audited report.

    Args:
        args_list (list, optional): Command line arguments. Defaults to None.
    """
    parser = argparse.ArgumentParser(description='Analyze tidal data')
    parser.add_argument('directory', help='Directory containing tidal data')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose')
    args, _ = parser.parse_known_args(args_list)

    folder = args.directory
    if not os.path.isdir(folder):
        sys.stderr.write(f"Error: Directory '{folder}' does not exist.\n")
        return

    sys.stderr.write(f"Scanning directory: {folder}\n")
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                   if f.endswith('.txt')])

    if not files:
        sys.stderr.write("Error: No .txt files found in directory.\n")
        return

    full_data = pd.DataFrame()
    try:
        sys.stderr.write(f"Loading {len(files)} files...\n")
        for f_path in files:
            full_data = pd.concat([full_data, read_tidal_data(f_path)])
    except (IOError, ValueError) as err:
        sys.stderr.write(f"Failed to process data files: {err}\n")
        return

    full_data = full_data.sort_index()
    full_data = full_data[~full_data.index.duplicated(keep='first')]

    sys.stderr.write("Computing analysis...\n")
    tides = calculate_tidal_components(full_data)
    regr = sea_level_rise(full_data)
    stretch = find_longest_contiguous_period(full_data)

    station = os.path.basename(os.path.normpath(folder)).capitalize()
    output = (
        f"Station: {station}\n"
        f"M2 amplitude: {tides[0]:.3f} m\n"
        f"S2 amplitude: {tides[1]:.3f} m\n"
        f"Sea-level rise: {regr[0] * 365.25:.4f} m/year\n"
        f"Longest contiguous period: {stretch} records\n"
        f"Total valid records: {len(full_data)}"
    )

    if args.verbose:
        print(output)
    else:
        report_file = f"{station.lower()}_report.txt"
        with open(report_file, "w", encoding="utf-8") as out_file:
            out_file.write(output)
        sys.stderr.write(f"Report saved to {report_file}\n")

    sys.stderr.write("Analysis complete.\n")

if __name__ == '__main__':
    main()
