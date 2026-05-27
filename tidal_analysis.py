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
    Reads tidal data, keeping Date/Time columns to satisfy test requirements.
    """
    data = pd.read_csv(
        file_path,
        sep=r'\s+',
        skiprows=11,
        names=['RowID', 'Date', 'Time', 'Sea Level', 'Residual'],
        usecols=['Date', 'Time', 'Sea Level']
    )

    # Convert to numeric
    data['Sea Level'] = pd.to_numeric(data['Sea Level'], errors='coerce')

    # Create the datetime index but DO NOT drop the original columns yet
    data['datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], errors='coerce')
    data.set_index('datetime', inplace=True)

    # Return Date, Time, and Sea Level to satisfy the test's .drop() requirement
    return data[['Date', 'Time', 'Sea Level']]

def extract_single_year_remove_mean(year, data):
    """
    Extracts data for a specific year and ensures exactly 8760/8784 rows.
    """
    # Create a full range of hourly timestamps for that year
    start_date = f"{year}-01-01 00:00:00"
    end_date = f"{year}-12-31 23:00:00"
    full_year_range = pd.date_range(start=start_date, end=end_date, freq='h')

    # Extract what we have and reindex to the full year
    # This fills in missing hours with NaN so the row count is perfect
    year_data = data.reindex(full_year_range)

    # Remove mean using only valid data
    annual_mean = year_data['Sea Level'].mean()
    year_data['Sea Level'] -= annual_mean

    # Name the index to match what some tests expect
    year_data.index.name = 'datetime'

    return year_data

def extract_section_remove_mean(start, end, data):
    """
    Extracts a specific time range and removes the mean.
    Ensures the end date includes the full day to match test expectations.
    """
    # Convert strings to datetime objects
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    # If the end date has no time component (it's 00:00:00),
    # extend it to the end of that day (23:00:00)
    if end_dt.hour == 0 and end_dt.minute == 0:
        end_dt = end_dt + pd.Timedelta(hours=23)

    # Create the full hourly range
    full_range = pd.date_range(start=start_dt, end=end_dt, freq='h')

    # Reindex fills in missing timestamps with NaN
    section = data.reindex(full_range)

    # Remove mean
    section_mean = section['Sea Level'].mean()
    section['Sea Level'] = section['Sea Level'] - section_mean

    # Set index name for consistency
    section.index.name = 'datetime'

    return section

def join_data(data1, data2):
    """
    Joins two dataframes and ensures the resulting index is a 
    contiguous hourly range from the earliest to the latest date.
    """
    # 1. Combine the dataframes
    joined_df = pd.concat([data1, data2])

    # 2. Sort and remove duplicates
    joined_df = joined_df.sort_index()
    joined_df = joined_df[~joined_df.index.duplicated(keep='first')]

    # 3. Create a full hourly range from the very start to the very end
    full_range = pd.date_range(start=joined_df.index.min(),
                               end=joined_df.index.max(),
                               freq='h')

    # 4. Reindex to ensure the row count matches exactly what the test expects
    joined_df = joined_df.reindex(full_range)
    joined_df.index.name = 'datetime'

    return joined_df

def sea_level_rise(data):
    """
    Calculates the slope of sea level rise in meters per day.
    """
    clean_data = data.dropna(subset=['Sea Level'])

    if len(clean_data) < 2:
        return 0.0, 1.0

    # Use matplotlib.dates to get a standardized float representation of days
    days = mdates.date2num(clean_data.index.to_pydatetime())

    slope, _, _, p_value, _ = linregress(days, clean_data['Sea Level'])

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
    # FIX: Drop NaNs here, right before the math, so we don't break uptide
    clean_data = data.dropna(subset=['Sea Level'])

    if clean_data.empty or len(clean_data) < 100:
        return 0.0, 0.0

    levels = clean_data['Sea Level'].values.astype(float)
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
    """
    Main entry point for the tidal analysis script.
    """
    parser = argparse.ArgumentParser(description='Analyze tidal data')
    parser.add_argument('directory', help='Directory containing tidal data')
    parser.add_argument('-v', '--verbose', action='store_true', help='Output to screen')
    args, _ = parser.parse_known_args(args_list)

    folder = args.directory
    sys.stderr.write(f"Scanning directory: {folder}\n")

    files = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                   if f.endswith('.txt')])

    if not files:
        sys.stderr.write("Error: No .txt files found in directory.\n")
        return

    sys.stderr.write(f"Loading {len(files)} files...\n")

    full_data = pd.DataFrame()
    for f_path in files:
        full_data = pd.concat([full_data, read_tidal_data(f_path)])

    full_data = full_data.sort_index()
    full_data = full_data[~full_data.index.duplicated(keep='first')]

    sys.stderr.write("Computing tidal analysis and regression...\n")

    # PACKING: We unpack directly into the f-string later or use results index
    tides = calculate_tidal_components(full_data)
    regr_results = sea_level_rise(full_data)
    longest_stretch = find_longest_contiguous_period(full_data)

    station = os.path.basename(os.path.normpath(folder)).capitalize()
    output = (
        f"Station: {station}\n"
        f"M2 amplitude: {tides[0]:.3f} m\n"
        f"S2 amplitude: {tides[1]:.3f} m\n"
        f"Sea-level rise: {regr_results[0] * 365.25:.4f} m/year\n"
        f"Longest contiguous period: {longest_stretch} records\n"
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
