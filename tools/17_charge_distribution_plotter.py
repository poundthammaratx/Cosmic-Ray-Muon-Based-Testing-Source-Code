# charge_distribution_plotter.py

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore") # Suppress unnecessary warnings
import pandas as pd
import glob
import sys
import os
import argparse
import re # Import for regular expressions (for more robust parsing)
from matplotlib.gridspec import GridSpec # Import for flexible subplot arrangement

# === START: Path and Import Management ===
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the 'util' subdirectory
util_path = os.path.join(script_dir, "util")
# Add the 'util' directory to sys.path so Python can find modules inside it
if util_path not in sys.path:
    sys.path.append(util_path) # FIX: Changed sys.sys.path to sys.path

# Import the dedicated HDF file reader function.
# Assuming 'hdf_reader.py' exists in 'util' and contains 'load_hdf_file_as_dict'.
try:
    from hdf_reader import load_hdf_file_as_dict
except ImportError:
    print(f"Error: Could not import 'load_hdf_file_as_dict' from 'hdf_reader.py' in {util_path}.")
    print("Please ensure 'hdf_reader.py' exists in the 'util' directory and contains the function.")
    sys.exit(1)

# Define 'read_hdffile' as an alias for our new, correct loading function.
read_hdffile = load_hdf_file_as_dict
# === END: Path and Import Management ===


# --- Start: Helper Functions ---

def linear_function(x, m, c):
    """Linear function for curve fitting: y = mx + c"""
    return m * x + c

def load_pmt_charges_as_dataframe(filename, min_charge_filter_pC=0.0):
    """
    Loads charge_ch1 and charge_ch2 from an HDF file, converts them to pC,
    and returns them as a pandas DataFrame. Filters out very low charges.

    Args:
        filename (str): Path to the HDF file.
        min_charge_filter_pC (float): Minimum charge (in pC) for an event to be included.
                                        Events with Ch1 OR Ch2 below this threshold will be filtered out.

    Returns:
        pd.DataFrame: DataFrame with 'ch1_pC' and 'ch2_pC' columns.
                      Returns an empty DataFrame if data is not found or cannot be processed.
    """
    data = read_hdffile(filename)
    if not data or "data" not in data or "metadata" not in data:
        warnings.warn(f"Missing 'data' or 'metadata' group in {filename}. Skipping.")
        return pd.DataFrame()

    try:
        raw_charges_ch1_adc = data["data"]["charge_ch1"][()]
        conversion_ch1 = data["metadata"]["conversion_ch1"][()]
        raw_charges_ch2_adc = data["data"]["charge_ch2"][()]
        conversion_ch2 = data["metadata"]["conversion_ch2"][()]
    except KeyError as e:
        warnings.warn(f"Missing expected HDF5 dataset in {filename}: {e}. Skipping.")
        return pd.DataFrame()
    except Exception as e:
        warnings.warn(f"An error occurred while accessing data in {filename}: {e}. Skipping.")
        return pd.DataFrame()

    # Conversion factor from your original analysis script
    conversion_factor_pC = (1e-6 * (1/60e6) * 1e12) # converts from (unit of conversion_ch) to pC

    charges_ch1_pC = raw_charges_ch1_adc * (conversion_ch1 * conversion_factor_pC)
    charges_ch2_pC = raw_charges_ch2_adc * (conversion_ch2 * conversion_factor_pC)

    # Ensure all arrays have the same length before creating DataFrame
    min_len = min(len(charges_ch1_pC), len(charges_ch2_pC))
    df = pd.DataFrame({
        'ch1_pC': charges_ch1_pC[:min_len],
        'ch2_pC': charges_ch2_pC[:min_len]
    })

    # Filter out very low charge events based on new min_charge_filter_pC
    # Using logical OR, so if EITHER channel's charge is above the threshold
    df = df[(df['ch1_pC'] > min_charge_filter_pC) | (df['ch2_pC'] > min_charge_filter_pC)]
    
    return df

def calculate_charge_correlation_slope(df_pmt_events, fit_q1_min_pC, fit_q1_max_pC):
    """
    Calculates the slope, intercept, and R-squared from the linear regression
    of Ch2 charge vs Ch1 charge within a specified Ch1 charge range.

    Args:
        df_pmt_events (pd.DataFrame): DataFrame containing 'ch1_pC' and 'ch2_pC'.
        fit_q1_min_pC (float): Minimum Ch1 charge (pC) for fitting.
        fit_q1_max_pC (float): Maximum Ch1 charge (pC) for fitting.

    Returns:
        tuple: (m_slope, c_intercept, r_squared). Returns NaN if fit fails or insufficient data.
    """
    m_slope, c_intercept, r_squared = np.nan, np.nan, np.nan
    
    # Filter data for fitting based on Ch1 charge range
    df_fit = df_pmt_events[(df_pmt_events['ch1_pC'] >= fit_q1_min_pC) & 
                           (df_pmt_events['ch1_pC'] <= fit_q1_max_pC)]

    if len(df_fit) >= 2: # Need at least 2 points for a linear fit
        try:
            # Fit the linear function with bounds to force slope between 0 and 1
            params, cov = curve_fit(linear_function, df_fit['ch1_pC'].values, df_fit['ch2_pC'].values,
                                    bounds=([0.0, -np.inf], [1.0, np.inf])) # 0 < m < 1
            m_slope, c_intercept = params
            
            # Calculate R-squared
            y_predicted = linear_function(df_fit['ch1_pC'].values, m_slope, c_intercept)
            ss_res = np.sum((df_fit['ch2_pC'].values - y_predicted)**2)
            ss_tot = np.sum((df_fit['ch2_pC'].values - np.mean(df_fit['ch2_pC'].values))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        except (RuntimeError, ValueError) as e:
            warnings.warn(f"Could not fit linear regression for Ch1 vs Ch2: {e}. Returning NaN for slope.")
    else:
        warnings.warn(f"Not enough valid charge data points in fit range [{fit_q1_min_pC:.1f}, {fit_q1_max_pC:.1f}] pC for correlation. Skipping regression.")
    
    return m_slope, c_intercept, r_squared

# --- End: Helper Functions ---


# --- Start: Plotting Functions ---

def plot_charge_distribution(charges_series, channel_label, unit_label, pmt_full_name, output_dir, pmt_output_prefix, x_min, x_max, bins_num, y_max_count, y_tick_increment, pc_per_pe_value=None):
    """
    Plots charge distribution (histogram) for a given channel in specified units.

    Args:
        charges_series (pd.Series): Series of charges (e.g., in pC or PE).
        channel_label (str): Label for the channel (e.g., "Ch1", "Ch2").
        unit_label (str): Unit of the charge (e.g., "pC", "PE").
        pmt_full_name (str): Full name of the PMT (e.g., "LOM16-01: PMT 00").
        output_dir (str): Directory to save the output PDF.
        pmt_output_prefix (str): Prefix for the output filename.
        x_min (float): Minimum value for the x-axis.
        x_max (float): Maximum value for the x-axis.
        bins_num (int): Number of bins for the histogram.
        y_max_count (float): Maximum value for the y-axis (counts).
        y_tick_increment (float): Increment for y-axis ticks.
        pc_per_pe_value (float, optional): If plotting Ch2 in PE, this is the pC/PE conversion factor.
                                            Used to display on the plot. Defaults to None.
    """
    charges_data = charges_series.values
    charges_data = charges_data[~np.isnan(charges_data)] # Remove NaNs

    if charges_data.size == 0:
        print(f"   No valid {channel_label} charge data for distribution for {pmt_full_name} in {unit_label}. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(x_min, x_max, bins_num + 1)
    ax.hist(charges_data, bins=bins, color='skyblue', edgecolor='blue', alpha=0.7)

    ax.set_xlabel(f'Charge ({unit_label})', fontsize=12)
    ax.set_ylabel('Counts', fontsize=12) # Always 'Counts'
    ax.set_title(f'Muon Charge Distribution: {channel_label} - {pmt_full_name}', fontsize=14)
    
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([0, y_max_count]) # Fixed Y-axis range
    ax.set_yticks(np.arange(0, y_max_count + y_tick_increment, y_tick_increment)) # Fixed Y-axis ticks
    ax.ticklabel_format(style='plain', axis='y') # Prevent scientific notation on y-axis for linear scale

    ax.tick_params(labelsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)

    # Add pC/PE conversion info for Ch2 PE plot
    if channel_label == "Ch2" and unit_label == "PE" and pc_per_pe_value is not None and not np.isnan(pc_per_pe_value):
        ax.text(0.95, 0.95, f'1 PE $\\approx$ {pc_per_pe_value:.3f} pC',
                transform=ax.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right', # Increased font size to 9
                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.6))

    output_filename = f"{pmt_output_prefix}{channel_label.lower().replace(' ', '')}_charge_distribution_{unit_label.lower()}.pdf"
    plt.savefig(os.path.join(output_dir, output_filename))
    plt.close(fig)
    print(f"Saved Muon Charge Distribution ({channel_label} in {unit_label}) plot for {pmt_full_name} to: {os.path.join(output_dir, output_filename)}")


def plot_multi_pmt_charge_distributions_overview(
    pmt_charges_data_collection, pmt_pc_per_pe_values, lom_name, output_dir, lom_output_prefix,
    channel_label, unit_label, x_min, x_max, bins_num, y_max_count, y_tick_increment
):
    """
    Plots charge distributions for all 16 PMTs of a LOM in a single 4x4 figure.

    Args:
        pmt_charges_data_collection (dict): Dictionary where keys are PMT IDs and values are
                                            pandas Series of charge data for that PMT.
        pmt_pc_per_pe_values (dict): Dictionary where keys are PMT IDs and values are
                                    the calculated pC/PE for Ch2 (only used for Ch2 PE plots).
        lom_name (str): Name of the LOM (e.g., "LOM16-01").
        output_dir (str): Directory to save the output PDF.
        lom_output_prefix (str): Prefix for the output filename.
        channel_label (str): Label for the channel (e.g., "Ch1", "Ch2").
        unit_label (str): Unit of the charge (e.g., "pC", "PE").
        x_min (float): Minimum value for the x-axis for all subplots.
        x_max (float): Maximum value for the x-axis for all subplots.
        bins_num (int): Number of bins for histograms in all subplots.
        y_max_count (float): Maximum value for the y-axis (counts) for all subplots.
        y_tick_increment (float): Increment for y-axis ticks for all subplots.
    """
    fig, axes = plt.subplots(4, 4, figsize=(20, 16)) # 4x4 grid for 16 subplots
    axes = axes.flatten()

    # Custom PMT order for plotting (matching physical layout if available)
    custom_subplot_order = np.array([8, 10, 12, 14, 0, 2, 4, 6, 1, 3, 5, 7, 9, 11, 13, 15]) 
    
    # Ensure all PMTs are included in a consistent order, using the actual data's PMT IDs
    pmt_ids_present = sorted(pmt_charges_data_collection.keys())
    pmt_ids_to_plot_ordered = [pid for pid in custom_subplot_order if pid in pmt_ids_present]
    if len(pmt_ids_to_plot_ordered) == 0 or len(pmt_ids_to_plot_ordered) < len(pmt_charges_data_collection): 
        pmt_ids_to_plot_ordered = sorted(pmt_charges_data_collection.keys()) # Use all available and sorted

    bins = np.linspace(x_min, x_max, bins_num + 1)

    for i, pmt_id in enumerate(pmt_ids_to_plot_ordered):
        ax = axes[i]
        charges_series = pmt_charges_data_collection.get(pmt_id)
        pmt_full_name = f"PMT {pmt_id:02d}"

        charges_data = None
        if charges_series is not None:
            charges_data = charges_series.values
            charges_data = charges_data[~np.isnan(charges_data)]

        if charges_data is not None and charges_data.size > 0:
            ax.hist(charges_data, bins=bins, color='skyblue', edgecolor='blue', alpha=0.7)
            ax.set_title(pmt_full_name, fontsize=14) # Increased font size for subplot title
            ax.set_xlabel(f'Charge ({unit_label})', fontsize=12) # Increased font size for x-label
            ax.set_ylabel('Counts', fontsize=12) # Increased font size for y-label
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([0, y_max_count]) # Fixed Y-axis range
            ax.set_yticks(np.arange(0, y_max_count + y_tick_increment, y_tick_increment)) # Fixed Y-axis ticks
            ax.ticklabel_format(style='plain', axis='y') # Prevent scientific notation on y-axis


            # Add pC/PE conversion info specifically for Ch2 PE plots
            if channel_label == "Ch2" and unit_label == "PE":
                pc_per_pe_val = pmt_pc_per_pe_values.get(pmt_id)
                if pc_per_pe_val is not None and not np.isnan(pc_per_pe_val) and pc_per_pe_val > 0:
                    ax.text(0.95, 0.95, f'1 PE $\\approx$ {pc_per_pe_val:.3f} pC',
                            transform=ax.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right', # Increased font size
                            bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.6))
                else:
                    ax.text(0.95, 0.95, 'Gain N/A',
                            transform=ax.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right', # Increased font size
                            bbox=dict(boxstyle='round,pad=0.2', fc='lightcoral', alpha=0.6))

        else:
            ax.text(0.5, 0.5, f"{pmt_full_name}\nNo Valid Data", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_title(pmt_full_name, fontsize=14) # Increased font size for subplot title
            ax.set_xticks([])
            ax.set_yticks([])

        ax.tick_params(labelsize=10) # Increased font size for tick labels on x/y axes
        ax.grid(True, linestyle=':', lw=0.5)

    fig.suptitle(f"{channel_label} Charge Distribution Overview - {lom_name} ({unit_label})", fontsize=20, y=1.0) # Increased font size for main title
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust rect for suptitle
    
    # Ensure correct output filename based on channel and unit
    output_filename = f"{lom_output_prefix}all_pmts_{channel_label.lower().replace(' ', '')}_distribution_overview_{unit_label.lower()}.pdf"
    plt.savefig(os.path.join(output_dir, output_filename))
    plt.close(fig)
    print(f"Saved all PMTs {channel_label} Distribution Overview ({unit_label}) plot for {lom_name} to: {os.path.join(output_dir, output_filename)}")


# --- End: Plotting Functions ---


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Charge Distributions (pC and PE) for PMT data.")
    parser.add_argument("lom_data_dirs", nargs='+', help="Input LOM data directories (e.g., 'path/to/LOM16-01/'). Each directory should contain 16 PMT HDF files.", type=str)
    parser.add_argument("--output_dir", "-o", help="Output base directory for plots (default: ./charge_distribution_plots)", type=str, default='./charge_distribution_plots')
    parser.add_argument("--assumed_ch1_pc_per_pe", type=float, default=0.8, help="Assumed pC per PE for Channel 1. Used for Ch1 PE conversion and Ch2 PE conversion via correlation slope. Default is 0.8 pC/PE.")
    parser.add_argument("--fit_q1_min", type=float, default=5.0, help="Minimum Ch1 charge (pC) for linear regression in Ch1-Ch2 correlation plot (used for Ch2 PE gain). Default 5.0 pC.")
    # Adjusted fit_q1_max to a higher value to better capture muon events for correlation
    parser.add_argument("--fit_q1_max", type=float, default=120.0, help="Maximum Ch1 charge (pC) for linear regression in Ch1-Ch2 correlation plot (used for Ch2 PE gain). Adjusted to 120.0 pC.")
    # New argument for the minimum charge filter to include very low charge events
    parser.add_argument("--min_charge_filter", type=float, default=0.05, help="Minimum charge (pC) for an event to be included in distribution plots. Lowering this can reveal SPE peak. Default is 0.05 pC.")
    args = parser.parse_args()

    # Constants for plotting ranges and bins
    # pC ranges
    CH_PC_X_MIN = 0
    CH_PC_X_MAX = 100
    CH_PC_BINS = 100 # 1 pC per bin for pC plots

    # PE ranges
    CH_PE_X_MIN = 0
    # Adjusted max for PE plots based on simulation image (around 200 PE)
    CH_PE_X_MAX = 220 
    CH_PE_BINS = 80 # Bins for PE plots

    # Y-axis constants for Counts
    Y_AXIS_MAX_COUNT = 300
    Y_AXIS_TICK_INCREMENT = 50

    # --- Create Output Directories ---
    output_base_dir = args.output_dir
    output_dirs = {
        # Individual PMT plots (now only linear scale)
        "ch1_pc": os.path.join(output_base_dir, "charge_distributions", "ch1_pC"),
        "ch2_pc": os.path.join(output_base_dir, "charge_distributions", "ch2_pC"),
        "ch1_pe": os.path.join(output_base_dir, "charge_distributions", "ch1_PE"),
        "ch2_pe": os.path.join(output_base_dir, "charge_distributions", "ch2_PE"),
        
        # Overview plots (16 PMTs in one PDF, now only linear scale)
        "ch1_pc_overview": os.path.join(output_base_dir, "charge_distributions_overview", "ch1_pC"),
        "ch2_pc_overview": os.path.join(output_base_dir, "charge_distributions_overview", "ch2_pC"),
        "ch1_pe_overview": os.path.join(output_base_dir, "charge_distributions_overview", "ch1_PE"),
        "ch2_pe_overview": os.path.join(output_base_dir, "charge_distributions_overview", "ch2_PE")
    }
    for d in output_dirs.values():
        os.makedirs(d, exist_ok=True)
    print(f"Output plots will be saved to: {output_base_dir} and its subfolders.")

    sorted_lom_data_dirs = sorted(args.lom_data_dirs)

    for lom_idx, current_lom_folder_path in enumerate(sorted_lom_data_dirs):
        current_lom_folder_path = os.path.normpath(current_lom_folder_path)
        current_lom_name = os.path.basename(current_lom_folder_path)

        print(f"\n========================================================")
        print(f"=== Processing LOM: {current_lom_name} ({lom_idx+1}/{len(sorted_lom_data_dirs)}) ===")
        print(f"========================================================")

        pmt_to_infile_map = {}
        current_lom_pmt_files = glob.glob(os.path.join(current_lom_folder_path, "*.hdf"))
        current_lom_pmt_files.sort()

        if not current_lom_pmt_files:
            print(f"Warning: No HDF files found directly in LOM folder {current_lom_folder_path}. Skipping this LOM.")
            continue

        # --- IMPORTANT FIX: Define lom_output_prefix here, outside the inner PMT loop ---
        # It needs to be available for the overview plots after all PMTs in this LOM are processed.
        # Try to parse run ID for prefix, otherwise use LOM name
        lom_runid = None
        # Check first file for run ID, assuming consistent across LOM
        if current_lom_pmt_files:
            first_file_base = os.path.basename(current_lom_pmt_files[0])
            run_match = re.search(r'run(\d+)', first_file_base, re.IGNORECASE)
            if run_match:
                lom_runid = int(run_match.group(1))

        if lom_runid is not None:
            lom_output_prefix = f"{current_lom_name}_Run{lom_runid}_"
        else:
            lom_output_prefix = f"{current_lom_name}_" # Fallback if runid not found
        # --- END IMPORTANT FIX ---

        for fname in current_lom_pmt_files:
            file_base_no_ext = os.path.splitext(os.path.basename(fname))[0]
            pmt_id_match = re.search(r'\.(\d{2})$', file_base_no_ext)
            if pmt_id_match:
                pmt_id = int(pmt_id_match.group(1))
                if 0 <= pmt_id <= 15:
                    pmt_to_infile_map[pmt_id] = fname
                else:
                    warnings.warn(f"Parsed PMT ID {pmt_id} from filename '{fname}' is out of expected range (0-15). Skipping.")
            else:
                warnings.warn(f"Could not parse PMT ID from filename '{fname}'. Skipping.")

        if not pmt_to_infile_map:
            print(f"No valid PMT files found in {current_lom_folder_path} to process. Skipping LOM.")
            continue

        pmt_ids_to_process_sorted = sorted(pmt_to_infile_map.keys())

        # Dictionaries to store data for multi-PMT overview plots
        pmt_ch1_pc_data_for_overview = {}
        pmt_ch2_pc_data_for_overview = {}
        pmt_ch1_pe_data_for_overview = {}
        pmt_ch2_pe_data_for_overview = {}
        # Dictionary to store calculated pC/PE values for Ch2 for display in overview plots
        pmt_ch2_pc_per_pe_for_overview_display = {}


        for pmt_id in pmt_ids_to_process_sorted:
            infile_pmt = pmt_to_infile_map[pmt_id]
            pmt_full_name = f"{current_lom_name}: PMT {pmt_id:02d}"
            pmt_output_prefix = f"{current_lom_name}_PMT{pmt_id:02d}_" # This prefix is for individual PMT plots


            print(f"\n--- Processing {pmt_full_name} ---")

            # Load charge data for the current PMT, applying the new min_charge_filter
            df_pmt_charges = load_pmt_charges_as_dataframe(infile_pmt, args.min_charge_filter)

            if df_pmt_charges.empty:
                print(f"   No valid charge data for {pmt_full_name}. Skipping plots for this PMT.")
                # Store empty series for overview plots to show "No Data"
                pmt_ch1_pc_data_for_overview[pmt_id] = pd.Series([], dtype=float) # Explicit dtype
                pmt_ch2_pc_data_for_overview[pmt_id] = pd.Series([], dtype=float) # Explicit dtype
                pmt_ch1_pe_data_for_overview[pmt_id] = pd.Series([], dtype=float) # Explicit dtype
                pmt_ch2_pe_data_for_overview[pmt_id] = pd.Series([], dtype=float) # Explicit dtype
                pmt_ch2_pc_per_pe_for_overview_display[pmt_id] = np.nan
                continue

            # --- Plot Individual Ch1 Charge Distribution in pC ---
            plot_charge_distribution(
                df_pmt_charges['ch1_pC'],
                "Ch1", "pC", pmt_full_name,
                output_dirs["ch1_pc"], pmt_output_prefix,
                CH_PC_X_MIN, CH_PC_X_MAX, CH_PC_BINS, 
                Y_AXIS_MAX_COUNT, Y_AXIS_TICK_INCREMENT # Added Y-axis params
            )
            pmt_ch1_pc_data_for_overview[pmt_id] = df_pmt_charges['ch1_pC'].copy()

            # --- Plot Individual Ch2 Charge Distribution in pC ---
            plot_charge_distribution(
                df_pmt_charges['ch2_pC'],
                "Ch2", "pC", pmt_full_name,
                output_dirs["ch2_pc"], pmt_output_prefix,
                CH_PC_X_MIN, CH_PC_X_MAX, CH_PC_BINS, 
                Y_AXIS_MAX_COUNT, Y_AXIS_TICK_INCREMENT # Added Y-axis params
            )
            pmt_ch2_pc_data_for_overview[pmt_id] = df_pmt_charges['ch2_pC'].copy()


            # --- Calculate slope for Ch2 PE conversion from Ch1-Ch2 correlation ---
            m_slope_ch2_vs_ch1, _, _ = calculate_charge_correlation_slope(
                df_pmt_charges, args.fit_q1_min, args.fit_q1_max
            )
            
            ch2_pc_per_pe_from_correlation = np.nan
            if not np.isnan(m_slope_ch2_vs_ch1) and args.assumed_ch1_pc_per_pe > 0:
                # Calculate pC/PE for Ch2 using the fitted slope and assumed Ch1 pC/PE
                ch2_pc_per_pe_from_correlation = m_slope_ch2_vs_ch1 * args.assumed_ch1_pc_per_pe
                print(f"    Calculated 1 PE for Ch2 (from correlation) = {ch2_pc_per_pe_from_correlation:.3f} pC")
            else:
                print(f"    Could not calculate Ch2 pC/PE from correlation for {pmt_full_name}. Ch2 PE plot may be skipped or invalid.")
            
            # Store for overview plot display
            pmt_ch2_pc_per_pe_for_overview_display[pmt_id] = ch2_pc_per_pe_from_correlation


            # --- Plot Individual Ch1 Charge Distribution in PE ---
            if args.assumed_ch1_pc_per_pe > 0:
                ch1_pe_charges = df_pmt_charges['ch1_pC'] / args.assumed_ch1_pc_per_pe
                plot_charge_distribution(
                    ch1_pe_charges,
                    "Ch1", "PE", pmt_full_name,
                    output_dirs["ch1_pe"], pmt_output_prefix,
                    CH_PE_X_MIN, CH_PE_X_MAX, CH_PE_BINS, 
                    Y_AXIS_MAX_COUNT, Y_AXIS_TICK_INCREMENT # Added Y-axis params
                )
                pmt_ch1_pe_data_for_overview[pmt_id] = ch1_pe_charges.copy()
            else:
                print(f"    Assumed Ch1 pC/PE is 0 or invalid. Skipping Ch1 PE distribution plot for {pmt_full_name}.")
                pmt_ch1_pe_data_for_overview[pmt_id] = pd.Series([], dtype=float) # Store empty Series


            # --- Plot Individual Ch2 Charge Distribution in PE (using calculated slope from correlation) ---
            if not np.isnan(ch2_pc_per_pe_from_correlation) and ch2_pc_per_pe_from_correlation > 0:
                ch2_pe_charges = df_pmt_charges['ch2_pC'] / ch2_pc_per_pe_from_correlation
                plot_charge_distribution(
                    ch2_pe_charges,
                    "Ch2", "PE", pmt_full_name,
                    output_dirs["ch2_pe"], pmt_output_prefix,
                    CH_PE_X_MIN, CH_PE_X_MAX, CH_PE_BINS, 
                    Y_AXIS_MAX_COUNT, Y_AXIS_TICK_INCREMENT, # Added Y-axis params
                    pc_per_pe_value=ch2_pc_per_pe_from_correlation # Pass for display on plot
                )
                pmt_ch2_pe_data_for_overview[pmt_id] = ch2_pe_charges.copy()
            else:
                print(f"    Calculated Ch2 pC/PE from correlation is invalid. Skipping Ch2 PE distribution plot for {pmt_full_name}.")
                pmt_ch2_pe_data_for_overview[pmt_id] = pd.Series([], dtype=float) # Store empty Series


        # --- Generate Multi-PMT Overview Plots for current LOM ---
        print(f"\n--- Generating Multi-PMT Overview Plots for {current_lom_name} ---")

        # Ch1 pC Overview
        plot_multi_pmt_charge_distributions_overview(
            pmt_ch1_pc_data_for_overview, {}, current_lom_name, # Empty dict for pC/PE values as not applicable
            output_dirs["ch1_pc_overview"], lom_output_prefix,
            "Ch1", "pC", CH_PC_X_MIN, CH_PC_X_MAX, CH_PC_BINS, 
            Y_AXIS_MAX_COUNT, Y_AXIS_TICK_INCREMENT # Added Y-axis params
        )

        # Ch2 pC Overview
        plot_multi_pmt_charge_distributions_overview(
            pmt_ch2_pc_data_for_overview, {}, current_lom_name, # Empty dict for pC/PE values as not applicable
            output_dirs["ch2_pc_overview"], lom_output_prefix,
            "Ch2", "pC", CH_PC_X_MIN, CH_PC_X_MAX, CH_PC_BINS, 
            Y_AXIS_MAX_COUNT, Y_AXIS_TICK_INCREMENT # Added Y-axis params
        )

        # Ch1 PE Overview
        # For Ch1 PE, the pc_per_pe_value is a global assumed_ch1_pc_per_pe, not per-PMT variable
        # So, we pass it as a dictionary where each PMT gets the same value for display purposes.
        ch1_pc_per_pe_display_dict = {pid: args.assumed_ch1_pc_per_pe for pid in pmt_ids_to_process_sorted}
        plot_multi_pmt_charge_distributions_overview(
            pmt_ch1_pe_data_for_overview, ch1_pc_per_pe_display_dict, current_lom_name,
            output_dirs["ch1_pe_overview"], lom_output_prefix,
            "Ch1", "PE", CH_PE_X_MIN, CH_PE_X_MAX, CH_PE_BINS, 
            Y_AXIS_MAX_COUNT, Y_AXIS_TICK_INCREMENT # Added Y-axis params
        )

        # Ch2 PE Overview (uses the calculated pC/PE values from correlation for display)
        plot_multi_pmt_charge_distributions_overview(
            pmt_ch2_pe_data_for_overview, pmt_ch2_pc_per_pe_for_overview_display, current_lom_name,
            output_dirs["ch2_pe_overview"], lom_output_prefix,
            "Ch2", "PE", CH_PE_X_MIN, CH_PE_X_MAX, CH_PE_BINS, 
            Y_AXIS_MAX_COUNT, Y_AXIS_TICK_INCREMENT # Added Y-axis params
        )

    print("\n========================================================")
    print("=== All Charge Distribution Plots Generated. ===")
    print("========================================================")