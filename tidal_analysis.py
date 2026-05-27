"""
Tidal Analysis Module
This script provides functions to read, join, and analyze sea-level data
using tidal constituent analysis.
"""
# Standard library imports
import argparse
import os
# Kept to prevent a NameError in test_tides.py due to a missing test dependency
import datetime  # pylint: disable=unused-import

# Third-party library imports
import numpy as np
import pandas as pd
import uptide
from scipy.stats import linregress

def read_tidal_data(file_path):
    # Using names that match your BODC image exactly
    data = pd.read_csv(
        file_path, 
        sep=r'\s+', 
        skiprows=11, 
        names=['RowID', 'Date', 'Time', 'Sea Level', 'Residual'],
        usecols=['Date', 'Time', 'Sea Level']
    )
    # Convert 'M', 'T', 'N' to NaN and drop them
    data['Sea Level'] = pd.to_numeric(data['Sea Level'], errors='coerce')
    data['datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], errors='coerce')
    data.set_index('datetime', inplace=True)
    return data.dropna(subset=['Sea Level'])

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
    Calculates the slope of sea level rise in meters per day.
    """
    # Ensure we aren't working with an empty slice
    clean_data = data.dropna(subset=['Sea Level'])

    if len(clean_data) < 2:
        # We need at least two points for a line
        return 0.0, 1.0

    # High precision time calculation
    # Using .index[0] on clean_data ensures we start at the first valid point
    t_start = clean_data.index[0]
    days = (clean_data.index - t_start).total_seconds() / 86400.0

    slope, _, _, _, p_value = linregress(days, clean_data['Sea Level'])

    return slope, p_value

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

def calculate_tidal_components(data):
    """
    Fits M2 and S2 tidal cycles to the data.
    """
    if data.empty or len(data) < 100:
        return 0.0, 0.0

    # Ensure we drop NaNs before extracting arrays
    clean_data = data.dropna(subset=['Sea Level'])
    if clean_data.empty:
        return 0.0, 0.0

    levels = clean_data['Sea Level'].values.astype(float)

    # Get the starting time (epoch) and convert the timeline to seconds
    t0 = clean_data.index[0]
    t_seconds = (clean_data.index - t0).total_seconds().values.astype(float)

    # Initialize the constituents
    tide = uptide.Tides(['M2', 'S2'])
    
    # FIX 1: Set the initial time so uptide can calculate astronomical positions
    tide.set_initial_time(t0.to_pydatetime())

    try:
        # Run the harmonic analysis
        coeffs = uptide.harmonic_analysis(tide, levels, t_seconds)

        # Handle different return formats depending on the uptide version
        if isinstance(coeffs, tuple):
            coeffs = coeffs[0]

        # FIX 2: Calculate the amplitude magnitude using np.abs
        m2_amp = float(np.abs(coeffs[0]))
        s2_amp = float(np.abs(coeffs[1]))

        return m2_amp, s2_amp

    except Exception as e:
        # FIX 3: Print the actual error to the console instead of silently returning 0
        print(f"Harmonic analysis failed: {e}")
        raise

def find_longest_contiguous_period(data):
    """
    Finds the longest stretch of data without gaps.
    """
    if data.empty:
        return 0
    # Create a boolean mask where True means the gap is standard (e.g., 15 or 60 mins)
    # We find where the time difference changes
    time_diffs = data.index.to_series().diff()
    # Any gap larger than the most common gap is a break
    is_break = time_diffs > time_diffs.median()
    # Group by the breaks and find the size of the largest group
    group_ids = is_break.cumsum()
    longest_period = data.groupby(group_ids).size().max()
    return longest_period

def main(args_list=None):
    parser = argparse.ArgumentParser(description='Analyze tidal data')
    parser.add_argument('directory', help='Directory containing tidal data')
    parser.add_argument('-v', '--verbose', action='store_true', help='Output to screen')
    args, _ = parser.parse_known_args(args_list)

    folder = args.directory
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.txt')])

    # Load and join all data for the station
    full_data = pd.DataFrame()
    for f in files:
        year_data = read_tidal_data(f)
        full_data = pd.concat([full_data, year_data])

    # Ensure it's sorted and no duplicates
    full_data = full_data.sort_index()
    full_data = full_data[~full_data.index.duplicated(keep='first')]

    # Perform Analysis
    m2, s2 = calculate_tidal_components(full_data)
    slope_per_day, _ = sea_level_rise(full_data)
    rise_per_year = slope_per_day * 365.25
    longest_stretch = find_longest_contiguous_period(full_data)

    # Format Output
    station_name = os.path.basename(os.path.normpath(folder)).capitalize()
    output = (
        f"Station: {station_name}\n"
        f"M2 amplitude: {m2:.3f} m\n"
        f"S2 amplitude: {s2:.3f} m\n"
        f"Sea-level rise: {rise_per_year:.4f} m/year\n"
        f"Longest contiguous period: {longest_stretch} records"
    )

    if args.verbose:
        print(output)
    else:
        # Save to file as per rules
        with open(f"{station_name.lower()}_report.txt", "w") as f:
            f.write(output)

if __name__ == '__main__':
    main()
