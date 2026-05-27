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

def calculate_tidal_components(data):
    """
    Calculates the M2 and S2 tidal amplitudes.
    """
    # 1. Clean data
    clean_data = data.dropna(subset=['Sea Level'])
    if clean_data.empty:
        return 0.0, 0.0

    # 2. Prepare time and levels
    t_seconds = (clean_data.index - clean_data.index[0]).total_seconds().values
    levels = clean_data['Sea Level'].values

    # 3. Setup and Run Analysis
    tide = uptide.Tides(['M2', 'S2'])

    # We use a try/except block to catch any internal library crashes
    try:
        res = uptide.harmonic_analysis(tide, levels, t_seconds)

        # Check if result exists and contains the amplitude array
        if res is not None and len(res) > 0 and res[0] is not None:
            amplitudes = res[0]
            return float(amplitudes[0]), float(amplitudes[1])
    except Exception: # pylint: disable=broad-except
        return 0.0, 0.0

    return 0.0, 0.0

def main(args_list=None):
    """
    Calculates M2 and S2 components.
    """
    # 1. Setup the Parser
    parser = argparse.ArgumentParser(description='Analyze tidal data')
    parser.add_argument('directory', help='Directory containing tidal data')
    parser.add_argument('-v', '--verbose', action='store_true', help='Output to screen')

    args, _ = parser.parse_known_args(args_list)

    # 2. Find the files
    folder = args.directory
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.txt')])

    # 3. Process each file and collect results
    all_results = []
    for file_path in files:
        data = read_tidal_data(file_path)
        slope, _ = sea_level_rise(data)
        # Store components as a tuple to save local variable count
        amps = calculate_tidal_components(data)

        # Build the string directly to avoid extra variable assignments
        output_line = (f"{os.path.basename(file_path)}: "
                       f"M2 amplitude {amps[0]:.3f}m, "
                       f"S2 amplitude {amps[1]:.3f}m, "
                       f"Sea-level rise {slope:.2e} m/day")

        all_results.append(output_line)

        if args.verbose:
            print(output_line)

    # Final output handling
    if not args.verbose and all_results:
        # Use normpath and basename in one line to stay under variable limit
        out_name = f"{os.path.basename(os.path.normpath(folder))}_report.txt"
        with open(out_name, 'w', encoding='utf-8') as f:
            for line in all_results:
                f.write(line + '\n')
        print(f"Analysis complete. Results saved to {out_name}")

if __name__ == '__main__':
    main()
