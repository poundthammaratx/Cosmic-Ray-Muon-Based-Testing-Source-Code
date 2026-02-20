import h5py
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore") # Suppress unnecessary warnings
import pandas as pd
import glob
import sys
import os
import json
import argparse
import scipy.fft # Import for Fast Fourier Transform
import re # Import for regular expressions (for more robust parsing)
from matplotlib.colors import LogNorm # For logarithmic color scale
from matplotlib.gridspec import GridSpec # For more flexible subplot arrangement

# === START: Path and Import Management ===
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the 'util' subdirectory
util_path = os.path.join(script_dir, "util")
# Add the 'util' directory to sys.path so Python can find modules inside it
if util_path not in sys.path:
    sys.path.append(util_path)

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


# Define the mapping for PMT physical positions to subplot indices in a 4x5 grid.
plotting_map = [6,11,7,12,8,13,9,14,1,16,2,17,3,18,4,19,5,20]


# --- Start: Helper Functions ---

# Define the linear function for curve_fit
def linear_function(x, m, c):
    return m * x + c

# Define a polynomial function for curve_fit (e.g., 2nd degree)
def polynomial_function(x, a, b, c):
    return a * x**2 + b * x + c

def get_timestamps(filename, min_charge_pC=0.0, max_charge_pC=float('inf')):
    """
    Retrieves FPGA timestamps and event indices for events from an HDF file.
    Events are selected if EITHER ch1 or ch2 charge falls within the specified charge range.

    Args:
        filename (str): Path to the HDF file.
        min_charge_pC (float): Minimum charge (in pC) for an event to be considered.
        max_charge_pC (float): Maximum charge (in pC) for an event to be considered.

    Returns:
        tuple: A tuple containing:
            - list: `ret_timestamps`, a list of FPGA timestamps (ns) for filtered events.
            - list: `ret_eventidx`, a list of original event indices for filtered events.
            Returns empty lists if data is not found or cannot be processed.
    """
    ret_timestamps=[]
    ret_eventidx=[]

    data = read_hdffile(filename)

    if not data or "data" not in data or "metadata" not in data:
        warnings.warn(f"Missing 'data' or 'metadata' group in {filename}. Skipping timestamp retrieval.")
        return [], []

    # Safely get data with checks for existence
    try:
        fpga_time = data["data"]["FPGAtime"][()]
        q_ch1_raw = data["data"]["charge_ch1"][()]
        conversion_ch1 = data["metadata"]["conversion_ch1"][()]
        q_ch2_raw = data["data"]["charge_ch2"][()]
        conversion_ch2 = data["metadata"]["conversion_ch2"][()]
    except KeyError as e:
        warnings.warn(f"Missing expected HDF5 dataset in {filename}: {e}. Skipping timestamp retrieval.")
        return [], []
    except Exception as e:
        warnings.warn(f"An error occurred while accessing data in {filename}: {e}. Skipping timestamp retrieval.")
        return [], []

    # Conversion factor from your 09_ultimate_muon_analysis.py
    conversion_factor_pC = (1e-6 * (1/60e6) * 1e12) # converts from (unit of conversion_ch) to pC

    q_ch1_pC = np.array([ s * (conversion_ch1 * conversion_factor_pC) for s in q_ch1_raw])
    q_ch2_pC = np.array([ s * (conversion_ch2 * conversion_factor_pC) for s in q_ch2_raw])

    # Ensure all arrays have the same length
    min_len = min(len(fpga_time), len(q_ch1_pC), len(q_ch2_pC))
    fpga_time = fpga_time[:min_len]
    q_ch1_pC = q_ch1_pC[:min_len]
    q_ch2_pC = q_ch2_pC[:min_len]


    for iev, (q1, q2, t) in enumerate(zip(q_ch1_pC, q_ch2_pC, fpga_time)):
        # Apply charge filtering: if EITHER channel's charge is in range
        if (min_charge_pC <= q1 <= max_charge_pC) or \
           (min_charge_pC <= q2 <= max_charge_pC):
            ret_timestamps.append(t)
            ret_eventidx.append(iev)

    return ret_timestamps, ret_eventidx


def event_matching(timestamp_ref_ns, timestamp_ns, window_ns=100.):
    """
    Finds matching timestamps between a reference set and a target set
    within a specified time window. This function uses a two-pointer approach
    and assumes both input timestamp lists are sorted.

    Args:
        timestamp_ref_ns (list or np.array): Sorted list of reference timestamps (ns).
        timestamp_ns (list or np.array): Sorted list of target timestamps (ns).
        window_ns (float): Time window (ns) for considering events as coincident.

    Returns:
        tuple: A tuple containing:
            - list: `ret_times`, a list of [t_ref, t_target] pairs for matched events.
            - int: `ngood`, the count of matched events.
    """
    ret_times=[]
    ngood=0

    timestamp_ref_ns = np.asarray(timestamp_ref_ns)
    timestamp_ns = np.asarray(timestamp_ns)

    # Handle empty input arrays
    if timestamp_ref_ns.size == 0 or timestamp_ns.size == 0:
        return [], 0

    ptr_ref, ptr_target = 0, 0
    while ptr_ref < len(timestamp_ref_ns) and ptr_target < len(timestamp_ns):
        t_ref = timestamp_ref_ns[ptr_ref]
        t_target = timestamp_ns[ptr_target]

        # Check if the events are within the coincidence window
        if abs(t_ref - t_target) <= window_ns:
            ret_times.append([t_ref, t_target])
            ngood += 1
            ptr_ref += 1
            ptr_target += 1
        # If reference timestamp is much greater than target, advance target pointer
        elif t_ref > t_target + window_ns: # t_target is too old
            ptr_target += 1
        # If target timestamp is much greater than reference, advance reference pointer
        else: # t_target < t_ref - window_ns (t_ref is too old)
            ptr_ref += 1

    return ret_times, ngood


def get_waveform_at_this_timestamp(filename, timestamp):
    """
    Retrieves the raw waveform (ADC counts) for a specific timestamp
    from a given HDF file, applying pedestal subtraction and Ch2 inversion,
    and specific display offsets for plotting.

    Args:
        filename (str): Path to the HDF file.
        timestamp (int): FPGA timestamp (ns) of the event to retrieve.

    Returns:
        tuple: A tuple containing:
            - np.array: `x`, time axis in nanoseconds (ns).
            - np.array: `wf_ch1`, preprocessed ADC counts for channel 1.
            - np.array: `wf_ch2`, preprocessed ADC counts for channel 2.
            Returns empty numpy arrays if no matching timestamp or data.
    """
    data = read_hdffile(filename)

    if not data or "data" not in data:
        warnings.warn(f"Missing 'data' group in {filename}. Cannot retrieve waveform.")
        return np.array([]), np.array([]), np.array([])

    fpga_time = data["data"]["FPGAtime"][()] if "FPGAtime" in data["data"] else np.array([])
    nsamples = data["data"]["nsample"][()] if "nsample" in data["data"] else np.array([])
    adc_ch1 = data["data"]["ADC_ch1"][()] if "ADC_ch1" in data["data"] else np.array([])
    adc_ch2 = data["data"]["ADC_ch2"][()] if "ADC_ch2" in data["data"] else np.array([])

    pedestal_ch1 = data["data"]["pedestal_ch1"][()] if "pedestal_ch1" in data["data"] else np.array([0])
    pedestal_ch2 = data["data"]["pedestal_ch2"][()] if "pedestal_ch2" in data["data"] else np.array([0])

    x = np.array([])
    wf_ch1 = np.array([])
    wf_ch2 = np.array([])

    matching_indices = np.where(fpga_time == timestamp)[0]
    if len(matching_indices) > 0:
        iev = matching_indices[0]

        if (iev < len(nsamples) and iev < adc_ch1.shape[0] and iev < adc_ch2.shape[0] and
            (pedestal_ch1.ndim == 0 or iev < len(pedestal_ch1)) and
            (pedestal_ch2.ndim == 0 or iev < len(pedestal_ch2))):

            n = nsamples[iev]

            x = np.array([ i * (1e9/60e6) for i in range(n) ])

            ped1_val = pedestal_ch1[iev] if pedestal_ch1.ndim > 0 and pedestal_ch1.size > iev else (pedestal_ch1.item() if pedestal_ch1.size > 0 else 0)
            ped2_val = pedestal_ch2[iev] if pedestal_ch2.ndim > 0 and pedestal_ch2.size > iev else (pedestal_ch2.item() if pedestal_ch2.size > 0 else 0)

            wf_ch1_raw = np.array(adc_ch1[iev][:n])
            wf_ch2_raw = np.array(adc_ch2[iev][:n])

            # --- Ch1 Processing (Pedestal Subtraction + Display Shift) ---
            wf_ch1 = wf_ch1_raw - ped1_val # Subtract pedestal (baseline near 0)
            wf_ch1 = wf_ch1 + 300 # Shift Ch1 waveform up by 300 (display offset)

            # --- Ch2 Processing: Use Raw ADC for Negative-Going Pulse Plot (as per user's image) ---
            # No inversion or major shifting here. The plotting function will set ylim and guide lines.
            wf_ch2 = wf_ch2_raw

        else:
            warnings.warn(f"Timestamp {timestamp} found, but associated event data at index {iev} is incomplete or out of bounds for {filename}. Skipping waveform retrieval.")

    return x, wf_ch1, wf_ch2


def get_charges_of_these_events(filename, evidx_list):
    """
    Retrieves charge values (in pC) for specific event indices from an HDF file.

    Args:
        filename (str): Path to the HDF file.
        evidx_list (list): List of event indices for which to retrieve charges.

    Returns:
        tuple: A tuple containing:
            - list: `ret_charges_ch1`, list of pC charges for channel 1.
            - list: `ret_charges_ch2`, list of pC charges for channel 2.
    """
    ret_charges_ch1=[]
    ret_charges_ch2=[]

    data = read_hdffile(filename)

    if not data or "data" not in data or "metadata" not in data:
        warnings.warn(f"Missing 'data' or 'metadata' group in {filename}. Skipping charge retrieval.")
        return [], []

    q_ch1 = data["data"]["charge_ch1"][()] if "charge_ch1" in data["data"] else np.array([])
    conversion_ch1 = data["metadata"]["conversion_ch1"][()] if "conversion_ch1" in data["metadata"] else 1.0
    q_ch2 = data["data"]["charge_ch2"][()] if "charge_ch2" in data["data"] else np.array([])
    conversion_ch2 = data["metadata"]["conversion_ch2"][()] if "conversion_ch2" in data["metadata"] else 1.0
    
    # Conversion factor from your 09_ultimate_muon_analysis.py
    conversion_factor_pC = (1e-6 * (1/60e6) * 1e12) # converts from (unit of conversion_ch) to pC

    q_ch1_pC_all = np.array([ s * (conversion_ch1 * conversion_factor_pC) for s in q_ch1])
    q_ch2_pC_all = np.array([ s * (conversion_ch2 * conversion_factor_pC) for s in q_ch2])

    max_charge_len = min(len(q_ch1_pC_all), len(q_ch2_pC_all))

    for iev in evidx_list:
        if iev < max_charge_len:
            ret_charges_ch1.append(q_ch1_pC_all[iev])
            ret_charges_ch2.append(q_ch2_pC_all[iev])
        else:
            ret_charges_ch1.append(np.nan)
            ret_charges_ch2.append(np.nan)
            warnings.warn(f"Event index {iev} out of bounds for charge data (length {max_charge_len}) in {filename}. Appending NaN.")

    return ret_charges_ch1, ret_charges_ch2
# --- End: Helper Functions ---


# --- Start: Plotting Functions ---

def plot_multi_pmt_waveforms_comparison(pmt_wf_data_collection, plot_title, output_filename_base, output_dir, x_range_mode="broad"):
    """
    Plots waveforms for multiple PMTs in a single figure with 4x4 subplots.
    Allows specifying the X-axis range mode.

    Args:
        pmt_wf_data_collection (dict): Dict like {pmt_id: {'x_wf': array, 'wf_data': array, 'ch_label': str, 'charge_pC': float}}.
        plot_title (str): Main title for the figure.
        output_filename_base (str): Base name for the output PDF file (e.g., "all_pmts_time_domain_ch1_comparison").
        output_dir (str): Directory to save the output PDF.
        x_range_mode (str): Defines the X-axis range. Can be "broad" (0-600ns) or "peak_focus".
                            "peak_focus" is (300-400ns for Ch1, 300-400ns for Ch2).
    """
    fig, axes = plt.subplots(4, 4, figsize=(15, 12)) # 4x4 grid for 16 subplots
    axes = axes.flatten()

    # Define the custom order of PMTs for the 4x4 subplot grid as requested
    custom_subplot_order_for_4x4_grid = np.array([
        8, 10, 12, 14,
        0,  2,  4,  6,
        1,  3,  5,  7,
        9, 11, 13, 15
    ])

    # --- Yuya's Suggestions for Y-limits ---
    global_ymin_adc_ch1 = 0
    global_ymax_adc_ch1 = 4095
    # Ch2: Negative going pulse, show baseline at 0 and above (as per image)
    global_ymin_adc_ch2 = 0
    global_ymax_adc_ch2 = 4095

    # --- Determine X-axis ranges based on x_range_mode ---
    if x_range_mode == "broad":
        xmin_mode = 0
        xmax_mode = 600
        filename_suffix = "_broad_view.pdf"
    elif x_range_mode == "peak_focus":
        xmin_mode = 300 # Yuya's suggestion for peak focus for BOTH Ch1 and Ch2
        xmax_mode = 400 # Yuya's suggestion for peak focus for BOTH Ch1 and Ch2
        filename_suffix = "_peak_focus.pdf"
    else:
        warnings.warn(f"Invalid x_range_mode: {x_range_mode}. Defaulting to broad view.")
        xmin_mode, xmax_mode = 0, 600
        filename_suffix = "_broad_view.pdf"

    output_filename = output_filename_base + filename_suffix


    # Plot each PMT's waveform
    for i, pmt_id_to_plot in enumerate(custom_subplot_order_for_4x4_grid):
        ax = axes[i]
        data = pmt_wf_data_collection.get(pmt_id_to_plot)

        # Set plot limits based on channel and selected x_range_mode
        if "Ch1" in plot_title: # This indicates we are plotting Ch1 data
            ymin_adc = global_ymin_adc_ch1
            ymax_adc = global_ymax_adc_ch1
        else: # This indicates we are plotting Ch2 data
            ymin_adc = global_ymin_adc_ch2
            ymax_adc = global_ymax_adc_ch2

        # X-limits are now common for both channels based on x_range_mode
        xmin_ns_current = xmin_mode
        xmax_ns_current = xmax_mode


        if data and data['wf_data'] is not None and len(data['wf_data']) > 0:
            ax.plot(data['x_wf'], data['wf_data'], color='blue')
            ax.set_title(f"PMT {pmt_id_to_plot} {data['ch_label']} ({data['charge_pC']:.1f}pC)", fontsize=8)
            ax.set_xlabel("Time (ns)", fontsize=7)
            ax.set_ylabel("ADC Counts", fontsize=7)
            ax.set_xlim([xmin_ns_current, xmax_ns_current])
            ax.set_ylim([ymin_adc, ymax_adc])

            # Add cutting line for Ch1 at ADC 300
            if "Ch1" in plot_title:
                ax.axhline(y=300, color='orange', linestyle='--', linewidth=0.8, alpha=0.7, label='ADC Cut (300)')
                ax.axhline(y=4095, color='purple', linestyle='-', linewidth=0.8, alpha=0.7, label='ADC Overflow Limit (4095)')
                ax.legend(fontsize=6, loc='upper right')
            # Add cutting lines for Ch2 based on new interpretation (raw ADC, high baseline, negative pulse)
            elif "Ch2" in plot_title:
                # Based on the image, the baseline is around 3800-3900 and pulse dips.
                # Assuming original pedestal value is around 3800-3900.
                ax.axhline(y=3800, color='gray', linestyle='-', linewidth=0.8, alpha=0.7, label='Baseline (~3800)')
                ax.axhline(y=3500, color='green', linestyle='--', linewidth=0.8, alpha=0.7, label='ADC Cut (3500)')
                # You can add more cut lines if desired, e.g., for lower values
                # ax.axhline(y=2000, color='red', linestyle='--', linewidth=0.8, alpha=0.7, label='ADC Cut (2000)')
                ax.legend(fontsize=6, loc='upper right')
        else:
            ax.text(0.5, 0.5, f"PMT {pmt_id_to_plot}\nNo Data in Peak", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_title(f"PMT {pmt_id_to_plot}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        ax.grid(True, linestyle=':', lw=0.5)
        ax.tick_params(labelsize=6)

    for j in range(len(custom_subplot_order_for_4x4_grid), 16):
        fig.delaxes(axes[j])

    fig.suptitle(plot_title + f" (X-Range: {x_range_mode.replace('_',' ').title()})", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, output_filename))
    plt.close(fig)
    print(f'Saved {os.path.join(output_dir, output_filename)}')


# Function to plot a multi-subplot comparison for FFTs
def plot_multi_pmt_ffts_comparison(pmt_fft_data_collection, plot_title, output_filename, output_dir):
    """
    Plots FFT magnitude spectra for multiple PMTs in a single figure with 4x4 subplots.
    pmt_fft_data_collection: Dict like {pmt_id: {'frequencies': array, 'magnitude': array, 'ch_label': str, 'charge_pC': float}}
    """
    fig, axes = plt.subplots(4, 4, figsize=(15, 12))
    axes = axes.flatten()

    custom_subplot_order_for_4x4_grid = np.array([
        8, 10, 12, 14,
        0,  2,  4,  6,
        1,  3,  5,  7,
        9, 11, 13, 15
    ])

    global_xmax_freq_mhz = float('-inf')
    for pmt_id in pmt_fft_data_collection.keys():
        data = pmt_fft_data_collection.get(pmt_id)
        if data and data['frequencies'] is not None and len(data['frequencies']) > 1:
            global_xmax_freq_mhz = max(global_xmax_freq_mhz, (data['frequencies'] / 1e6)[-1])
    if global_xmax_freq_mhz == float('-inf'): global_xmax_freq_mhz = 30

    for i, pmt_id_to_plot in enumerate(custom_subplot_order_for_4x4_grid):
        ax = axes[i]
        data = pmt_fft_data_collection.get(pmt_id_to_plot)

        if data and data['magnitude'] is not None and len(data['magnitude']) > 0:
            ax.plot(data['frequencies'] / 1e6, data['magnitude'], color='red')
            ax.set_title(f"PMT {pmt_id_to_plot} {data['ch_label']} ({data['charge_pC']:.1f}pC)", fontsize=8)
            ax.set_xlabel("Frequency (MHz)", fontsize=7)
            ax.set_ylabel("Magnitude", fontsize=7)
            ax.set_xlim([0, global_xmax_freq_mhz])
            ax.set_yscale('log')
        else:
            ax.text(0.5, 0.5, f"PMT {pmt_id_to_plot}\nNo Data in Peak", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_title(f"PMT {pmt_id_to_plot}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        ax.grid(True, linestyle=':', lw=0.5)
        ax.tick_params(labelsize=6)

    for j in range(len(custom_subplot_order_for_4x4_grid), 16):
        fig.delaxes(axes[j])

    fig.suptitle(plot_title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, output_filename))
    plt.close(fig)
    print(f'Saved {os.path.join(output_dir, output_filename)}')


# NEW FUNCTION: plot_muon_charge_distribution_ch2
def plot_muon_charge_distribution_ch2(pmt_to_infile_map, assumed_gain, output_dir_dist, lom_output_prefix=""):
    all_ch2_charges_pe = []
    electron_charge_C = 1.602e-19
    charge_per_pe_C = assumed_gain * electron_charge_C
    charge_per_pe_pC = charge_per_pe_C * 1e12 # pC per PE

    print("\n--- Generating Muon Charge Distribution (Ch2) ---")
    for pmt_id, infile_path in pmt_to_infile_map.items():
        data = read_hdffile(infile_path)
        if not data or "data" not in data or "metadata" not in data:
            warnings.warn(f"Missing 'data' or 'metadata' group in {infile_path}. Skipping charge distribution for PMT {pmt_id}.")
            continue

        if "charge_ch2" in data["data"] and "conversion_ch2" in data["metadata"]:
            raw_charges_ch2_adc = data["data"]["charge_ch2"][()]
            # Re-using conversion_factor_pC from get_timestamps for consistency
            conversion_ch2 = data["metadata"]["conversion_ch2"][()]
            conversion_factor_pC = (1e-6 * (1/60e6) * 1e12) # Re-use consistent conversion
            charges_pC = raw_charges_ch2_adc * (conversion_ch2 * conversion_factor_pC)
            charges_pe = charges_pC / charge_per_pe_pC

            all_ch2_charges_pe.extend(charges_pe)
        else:
            warnings.warn(f"Missing 'charge_ch2' or 'conversion_ch2' in {infile_path}. Skipping charge distribution for PMT {pmt_id}.")

    if not all_ch2_charges_pe:
        print("No Ch2 charge data found across all PMTs to plot distribution.")
        return

    all_ch2_charges_pe = np.array(all_ch2_charges_pe)
    all_ch2_charges_pe = all_ch2_charges_pe[~np.isnan(all_ch2_charges_pe)]

    fig, ax = plt.subplots(figsize=(10, 6))

    # MODIFIED: Adjusted X-axis range and Y-axis limit for distribution plot
    x_dist_min_pe = 0
    x_dist_max_pe = 20
    y_dist_min_counts = 0
    y_dist_max_counts = 4500
    bins = np.linspace(x_dist_min_pe, x_dist_max_pe, 81) # 0.25 PE bin size for 0-20 PE

    ax.hist(all_ch2_charges_pe, bins=bins, color='skyblue', edgecolor='blue', alpha=0.7)

    ax.set_xlabel('Charge (PE)', fontsize=12)
    ax.set_ylabel('Counts (Linear Scale)', fontsize=12)
    ax.set_title(f'Muon Charge Distribution: Channel 2 (All PMTs Combined) - {lom_output_prefix.strip("_")}', fontsize=14)
    
    ax.set_xlim([x_dist_min_pe, x_dist_max_pe]) # MODIFIED: X-axis range 0-20 PE
    ax.set_ylim([y_dist_min_counts, y_dist_max_counts]) # MODIFIED: Y-axis range 0-4500 counts
    ax.set_yscale('linear') # Ensure linear scale as requested

    ax.tick_params(labelsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)

    output_filename = f"{lom_output_prefix}muon_charge_distribution_ch2_linear_scale.pdf"
    plt.savefig(os.path.join(output_dir_dist, output_filename))
    plt.close(fig)
    print(f"Saved Muon Charge Distribution (Ch2) plot to: {os.path.join(output_dir_dist, output_filename)}")


# New function to plot a single charge correlation subplot
def plot_single_charge_correlation_subplot(ax, all_events_df, lom_name, m_slope, c_intercept, r_squared, fit_q1_min_pC, fit_q1_max_pC, plot_type='log'):
    x_data_all = all_events_df['ch1_pC'].values
    y_data_all = all_events_df['ch2_pC'].values

    # Define common ranges for these plots as per Chris Wendt's suggestion
    x_plot_min_pC = 0
    x_plot_max_pC = 100
    y_plot_min_pC = 0
    y_plot_max_pC = 10

    # Determine bins for the 2D histograms (for the new focused ranges)
    x_bins_focused = np.linspace(x_plot_min_pC, x_plot_max_pC, 101) # 1 pC bin size
    y_bins_focused = np.linspace(y_plot_min_pC, y_plot_max_pC, 51)  # 0.2 pC bin size

    if plot_type == 'log':
        h = ax.hist2d(x_data_all, y_data_all, bins=[x_bins_focused, y_bins_focused], cmap='viridis', norm=LogNorm())
        # We don't add the colorbar here, we'll handle it outside for combined plots
        title_suffix = '(Log Scale)'
    else: # linear
        h = ax.hist2d(x_data_all, y_data_all, bins=[x_bins_focused, y_bins_focused], cmap='viridis')
        # We don't add the colorbar here, we'll handle it outside for combined plots
        title_suffix = '(Linear Scale)'
    
    ax.set_title(f'LOM: {lom_name} {title_suffix}', fontsize=10) # Smaller title for subplots
    ax.set_xlabel('Charge Ch1 (pC)', fontsize=8) # Smaller labels
    ax.set_ylabel('Charge Ch2 (pC)', fontsize=8)
    ax.set_xlim(x_plot_min_pC, x_plot_max_pC)
    ax.set_ylim(y_plot_min_pC, y_plot_max_pC)
    ax.tick_params(labelsize=7) # Smaller tick labels
    ax.grid(True, linestyle=':', alpha=0.6)

    # Plot the regression line
    if not np.isnan(m_slope):
        x_fit_plot = np.linspace(x_plot_min_pC, x_plot_max_pC, 100)
        y_fit_plot = linear_function(x_fit_plot, m_slope, c_intercept)
        ax.plot(x_fit_plot, y_fit_plot, color='red', linestyle='--', linewidth=1.5,
                             label=f'Fit: Q2 = {m_slope:.2f}*Q1 + {c_intercept:.2f} pC\n($R^2={r_squared:.2f}$)\nFit Range: [{fit_q1_min_pC:.1f}-{fit_q1_max_pC:.1f}] pC')
    ax.legend(fontsize=6, loc='upper left') # Smaller legend

    return h[3] # Return the mappable for potential common colorbar


def plot_sample_waveforms(selected_events, plot_title, output_filename, output_dir, x_range_mode="broad"):
    """
    Plots waveforms for a selection of specific events in a grid.

    Args:
        selected_events (list of dict): List of dictionaries, each with keys:
            'filename', 'timestamp', 'pmt_id', 'ch1_pC', 'ch2_pC', 'event_type' (e.g., 'well-correlated', 'outlier').
        plot_title (str): Main title for the figure.
        output_filename (str): Name for the output PDF file.
        output_dir (str): Directory to save the output PDF.
        x_range_mode (str): Defines the X-axis range. Can be "broad" (0-600ns) or "peak_focus".
    """
    if not selected_events:
        print(f"No events provided for plotting: {plot_title}. Skipping.")
        return

    # Determine grid size (e.g., 2xN or 3xN, max 16 plots)
    num_plots = len(selected_events)
    if num_plots == 0: return
    
    ncols = min(4, num_plots)
    nrows = (num_plots + ncols - 1) // ncols # Ceiling division

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    axes = axes.flatten()

    # --- Determine X-axis ranges based on x_range_mode ---
    if x_range_mode == "broad":
        xmin_mode = 0
        xmax_mode = 600
    elif x_range_mode == "peak_focus":
        xmin_mode = 300
        xmax_mode = 400
    else:
        warnings.warn(f"Invalid x_range_mode: {x_range_mode}. Defaulting to broad view.")
        xmin_mode, xmax_mode = 0, 600

    global_ymin_adc_ch1 = 0
    global_ymax_adc_ch1 = 4095
    global_ymin_adc_ch2 = 0
    global_ymax_adc_ch2 = 4095

    for i, event_info in enumerate(selected_events):
        if i >= len(axes): # Prevent plotting more than available subplots
            break
        ax = axes[i]
        
        filename = event_info['filename']
        timestamp = event_info['timestamp']
        pmt_id = event_info['pmt_id']
        ch1_pC = event_info['ch1_pC']
        ch2_pC = event_info['ch2_pC']
        event_type = event_info.get('event_type', 'Event') # Default to 'Event'

        x_wf, wf1_data, wf2_data = get_waveform_at_this_timestamp(filename, timestamp)

        if len(x_wf) > 0:
            # Plot Ch1
            ax.plot(x_wf, wf1_data, color='blue', label='Ch1 (pedestal subtracted, shifted)')
            # Plot Ch2 (raw ADC)
            ax.plot(x_wf, wf2_data, color='green', label='Ch2 (raw ADC)')

            ax.set_title(f"PMT {pmt_id} ({event_type})\nCh1:{ch1_pC:.1f}pC, Ch2:{ch2_pC:.1f}pC", fontsize=8)
            ax.set_xlabel("Time (ns)", fontsize=7)
            ax.set_ylabel("ADC Counts", fontsize=7)
            ax.set_xlim([xmin_mode, xmax_mode])
            # Set Y-limits to encompass both channels' typical ranges
            ax.set_ylim(min(global_ymin_adc_ch1, global_ymin_adc_ch2), max(global_ymax_adc_ch1, global_ymax_adc_ch2))

            # Add cutting lines for Ch1 and Ch2
            ax.axhline(y=300, color='orange', linestyle=':', linewidth=0.8, alpha=0.7, label='Ch1 ADC Cut (300)')
            ax.axhline(y=3800, color='gray', linestyle=':', linewidth=0.8, alpha=0.7, label='Ch2 Baseline (~3800)')
            ax.axhline(y=3500, color='purple', linestyle=':', linewidth=0.8, alpha=0.7, label='Ch2 ADC Cut (3500)')
            ax.legend(fontsize=5, loc='upper right')

        else:
            ax.text(0.5, 0.5, f"PMT {pmt_id}\nNo Waveform Data", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_title(f"PMT {pmt_id} ({event_type})", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        ax.grid(True, linestyle=':', lw=0.5)
        ax.tick_params(labelsize=6)

    # Hide any unused subplots
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(plot_title + f" (X-Range: {x_range_mode.replace('_',' ').title()})", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, output_filename))
    plt.close(fig)
    print(f'Saved {os.path.join(output_dir, output_filename)}')


# --- End: Plotting Functions ---


if __name__ == "__main__":
    # Configure command-line argument parser
    parser = argparse.ArgumentParser(description="Streamlined Muon Analysis Script: Focuses on coincidence and FFT analysis.")
    parser.add_argument("lom_data_dirs", nargs='+', help="Input LOM data directories (e.g., 'path/to/LOM16-01/'). Each directory should contain 16 PMT HDF files.", type=str)
    parser.add_argument("--output_dir", "-o", help="Output directory for plots (default: ./plots_output)", type=str, default='./plots_output')
    parser.add_argument("--assumed_gain", type=float, default=5e6, help="Assumed PMT gain for PE conversion (electrons/PE). Default is 5e6.")
    parser.add_argument("--fit_q1_min", type=float, default=5.0, help="Minimum Ch1 charge (pC) for linear regression in correlation plot.")
    parser.add_argument("--fit_q1_max", type=float, default=100.0, help="Maximum Ch1 charge (pC) for linear regression in correlation plot.")
    parser.add_argument("--waveform_sample_size", type=int, default=5, help="Number of well-correlated and outlier waveforms to sample and plot.")
    args = parser.parse_args()

    # Define the fixed time window as requested by Yuya
    # This will be used for all coincidence calculations
    FIXED_COINCIDENCE_WINDOW_NS = 100.0

    # Define the coincidence thresholds as before
    COINCIDENCE_THRESHOLDS_PC = [10.0, 20.0, 50.0, 100.0]

    # Assumed pC/PE for Channel 1 (given by Yuya)
    ASSUMED_CH1_PC_PER_PE = 0.8

    # --- Create Output Directories ---
    output_base_dir = args.output_dir
    os.makedirs(output_base_dir, exist_ok=True)

    output_dirs = {
        "coincidence_matrices": os.path.join(output_base_dir, "coincidence_matrices"),
        "coincidence_matrices_log": os.path.join(output_base_dir, "coincidence_matrices_log"), # NEW: for log scale matrices
        "coincidence_rates": os.path.join(output_base_dir, "coincidence_rates"),
        "coincidence_rates_log": os.path.join(output_base_dir, "coincidence_rates_log"), # NEW: for log scale rates
        "waveform_analysis_ch1": os.path.join(output_base_dir, "waveform_analysis_ch1"),
        "waveform_analysis_ch2": os.path.join(output_base_dir, "waveform_analysis_ch2"),
        "fft_analysis_ch1": os.path.join(output_base_dir, "fft_analysis_ch1"),
        "fft_analysis_ch2": os.path.join(output_base_dir, "fft_analysis_ch2"),
        "charge_distributions": os.path.join(output_base_dir, "charge_distributions"),
        "charge_correlations": os.path.join(output_base_dir, "charge_correlations"),
        "waveform_samples": os.path.join(output_base_dir, "waveform_samples") # NEW: For sampled waveforms
    }
    for d in output_dirs.values():
        os.makedirs(d, exist_ok=True)
    print(f"Output plots will be saved to: {output_base_dir} and its subfolders.")

    # --- Start: Main loop for LOM-by-LOM processing ---
    # Sort LOM directories for consistent processing order
    sorted_lom_data_dirs = sorted(args.lom_data_dirs)
    num_loms = len(sorted_lom_data_dirs)
    
    # --- Prepare combined figures for multiple LOMs ---
    # Determine grid size (e.g., 2xN or 3xN, max 16 plots)
    
    # --- MODIFIED: Charge Correlation combined figure layout and colorbars ---
    # New layout: 2 rows (Linear, Log), num_loms columns for plots
    # Plus an extra column on the far right for two global colorbars.
    gs_corr = GridSpec(2, num_loms + 1, width_ratios=[1]*num_loms + [0.05], wspace=0.1, hspace=0.3)
    fig_corr_combined = plt.figure(figsize=(num_loms * 4.5 + 2, 10)) # Adjusted figsize for landscape and cbar
    fig_corr_combined.suptitle(f"Charge Correlation (Ch1 vs Ch2) for All LOMs", fontsize=18, y=0.99)
    
    # Lists to collect mappables for common colorbars
    linear_corr_mappables = [] 
    log_corr_mappables = []

    # Coincidence Matrices combined figures (remains 2 rows for each scale)
    num_thresholds = len(COINCIDENCE_THRESHOLDS_PC)
    
    # Linear Scale Coincidence Matrices for all LOMs
    # Adjusted figsize for landscape orientation and more space for colorbars
    gs_matrices_linear = GridSpec(num_loms, num_thresholds + 1, width_ratios=[1]*num_thresholds + [0.05], wspace=0.1, hspace=0.3)
    fig_matrices_all_loms_linear = plt.figure(figsize=(6 * num_thresholds + 3, 5.5 * num_loms)) # Added more space for cbar
    fig_matrices_all_loms_linear.suptitle(f"Coincidence Rate Matrices (Linear Scale) for All LOMs\nFixed ΔT: {FIXED_COINCIDENCE_WINDOW_NS} ns", fontsize=16, y=0.99)
    matrices_linear_mappables = []

    # Log Scale Coincidence Matrices for all LOMs
    # Adjusted figsize for landscape orientation and more space for colorbars
    gs_matrices_log = GridSpec(num_loms, num_thresholds + 1, width_ratios=[1]*num_thresholds + [0.05], wspace=0.1, hspace=0.3)
    fig_matrices_all_loms_log = plt.figure(figsize=(6 * num_thresholds + 3, 5.5 * num_loms)) # Added more space for cbar
    fig_matrices_all_loms_log.suptitle(f"Coincidence Rate Matrices (Log Scale) for All LOMs\nFixed ΔT: {FIXED_COINCIDENCE_WINDOW_NS} ns", fontsize=16, y=0.99)
    matrices_log_mappables = []

    # Define fixed colorbar ticks for linear scale (for coincidence matrices)
    fixed_cbar_ticks = np.array([0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00])
    fixed_cbar_vmax = fixed_cbar_ticks[-1] # Max value for colorbar


    for lom_idx, current_lom_folder_path in enumerate(sorted_lom_data_dirs):
        # Normalize path to remove trailing slashes for clean basename extraction
        current_lom_folder_path = os.path.normpath(current_lom_folder_path)
        current_lom_name = os.path.basename(current_lom_folder_path)

        print(f"\n========================================================")
        print(f"=== Processing LOM: {current_lom_name} ({lom_idx+1}/{len(sorted_lom_data_dirs)}) ===")
        print(f"========================================================")

        # Reset map and lists for the current LOM's data
        pmt_to_infile_map = {}
        current_lom_runid = None # Reset runid per LOM

        # Find all HDF files within this specific LOM folder
        current_lom_pmt_files = glob.glob(os.path.join(current_lom_folder_path, "*.hdf"))
        current_lom_pmt_files.sort() # Ensure consistent order of PMT files

        if not current_lom_pmt_files:
            print(f"Warning: No HDF files found directly in LOM folder {current_lom_folder_path}. Skipping this LOM.")
            continue # Skip to next LOM if no files

        # Populate pmt_to_infile_map and parse runid for the current LOM
        for fname in current_lom_pmt_files:
            file_base = os.path.basename(fname)
            file_base_no_ext = os.path.splitext(file_base)[0] # e.g., "data-muon-run909.00"

            pmt_id = -1 # Default invalid ID

            # Robust PMT ID Parsing from "data-muon-run909.XX" format
            pmt_id_match = re.search(r'\.(\d{2})$', file_base_no_ext) # Regex to find .XX at the very end
            if pmt_id_match:
                pmt_id = int(pmt_id_match.group(1))

            if 0 <= pmt_id <= 15: # Assuming PMT IDs are 0-15
                pmt_to_infile_map[pmt_id] = fname
            else:
                warnings.warn(f"Parsed PMT ID {pmt_id} from filename '{fname}' is out of expected range (0-15) or invalid. Skipping for PMT map for LOM {current_lom_name}.")

            # Extract Run ID from file name (e.g., 'data-muon-run909.00.hdf' -> 909)
            run_match = re.search(r'run(\d+)', file_base, re.IGNORECASE)
            if run_match:
                current_runid_candidate = int(run_match.group(1))
                if current_lom_runid is None:
                    current_lom_runid = current_runid_candidate
                elif current_lom_runid != current_runid_candidate:
                    warnings.warn(f'Run ID inconsistency within LOM {current_lom_name}: {current_runid_candidate} vs {current_lom_runid}. Using {current_lom_runid}.' )
            else:
                warnings.warn(f"Could not parse Run ID from filename '{fname}' for LOM {current_lom_name}.")


        if not pmt_to_infile_map or len(pmt_to_infile_map) < 16: # Warn if not all 16 PMTs found for this LOM
            warnings.warn(f"Warning: Only {len(pmt_to_infile_map)} PMT files found for LOM {current_lom_name}. Coincidence/FFT plots may be incomplete or skipped.")
            # Decide if to skip entire LOM if not all 16 PMTs are found. For now, allow partial.

        # Prepare LOM-specific prefix for output filenames
        lom_output_prefix = current_lom_name
        if current_lom_runid is not None:
            lom_output_prefix = f"{current_lom_name}_Run{current_lom_runid}_"
        else:
            lom_output_prefix = f"{current_lom_name}_" # Fallback if runid not found

        # --- Charge Correlation subplot for current LOM ---
        all_events_data_current_lom = []
        for pmt_id, infile_path in pmt_to_infile_map.items():
            data = read_hdffile(infile_path)
            if not data or "data" not in data or "metadata" not in data:
                continue # Skip PMT if data missing
            try:
                raw_charges_ch1_adc = data["data"]["charge_ch1"][()]
                conversion_ch1 = data["metadata"]["conversion_ch1"][()]
                raw_charges_ch2_adc = data["data"]["charge_ch2"][()]
                conversion_ch2 = data["metadata"]["conversion_ch2"][()]
                fpga_time = data["data"]["FPGAtime"][()]
            except KeyError as e:
                warnings.warn(f"Missing charge, conversion, or FPGAtime data for PMT {pmt_id} in {infile_path}: {e}. Skipping for correlation.")
                continue
            
            conversion_factor_pC = (1e-6 * (1/60e6) * 1e12) # consistent conversion factor

            charges_ch1_pC = raw_charges_ch1_adc * (conversion_ch1 * conversion_factor_pC)
            charges_ch2_pC = raw_charges_ch2_adc * (conversion_ch2 * conversion_factor_pC)

            min_len = min(len(charges_ch1_pC), len(charges_ch2_pC), len(fpga_time))
            for i in range(min_len):
                all_events_data_current_lom.append({
                    'ch1_pC': charges_ch1_pC[i],
                    'ch2_pC': charges_ch2_pC[i],
                    'filename': infile_path,
                    'timestamp': fpga_time[i],
                    'event_idx': i,
                    'pmt_id': pmt_id
                })

        df_all_events = pd.DataFrame(all_events_data_current_lom)
        df_all_events = df_all_events.dropna(subset=['ch1_pC', 'ch2_pC'])
        df_all_events = df_all_events[(df_all_events['ch1_pC'] > 1) & (df_all_events['ch2_pC'] > 1)]

        m_slope, c_intercept, r_squared = np.nan, np.nan, np.nan
        df_fit = df_all_events[(df_all_events['ch1_pC'] >= args.fit_q1_min) & (df_all_events['ch1_pC'] <= args.fit_q1_max)]

        if len(df_fit) >= 2:
            try:
                params, cov = curve_fit(linear_function, df_fit['ch1_pC'].values, df_fit['ch2_pC'].values)
                m_slope, c_intercept = params
                residuals = df_fit['ch2_pC'].values - linear_function(df_fit['ch1_pC'].values, m_slope, c_intercept)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((df_fit['ch2_pC'].values - np.mean(df_fit['ch2_pC'].values))**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
            except RuntimeError:
                warnings.warn(f"Could not fit linear regression for LOM {current_lom_name}.")
        else:
            warnings.warn(f"Not enough valid charge data points in fit range [{args.fit_q1_min:.1f}, {args.fit_q1_max:.1f}] pC for LOM {current_lom_name}. Skipping regression.")

        # --- MODIFIED: Plotting Linear Scale subplot for current LOM (top row) ---
        ax_linear_corr = fig_corr_combined.add_subplot(gs_corr[0, lom_idx])
        mappable_linear = plot_single_charge_correlation_subplot(
            ax_linear_corr, df_all_events, current_lom_name,
            m_slope, c_intercept, r_squared, args.fit_q1_min, args.fit_q1_max, plot_type='linear'
        )
        if mappable_linear: linear_corr_mappables.append(mappable_linear)

        # --- MODIFIED: Plotting Log Scale subplot for current LOM (bottom row) ---
        ax_log_corr = fig_corr_combined.add_subplot(gs_corr[1, lom_idx])
        mappable_log = plot_single_charge_correlation_subplot(
            ax_log_corr, df_all_events, current_lom_name,
            m_slope, c_intercept, r_squared, args.fit_q1_min, args.fit_q1_max, plot_type='log'
        )
        if mappable_log: log_corr_mappables.append(mappable_log)
        
        # --- NEW: Separate and Analyze Event Populations for the current LOM ---
        if not np.isnan(m_slope) and not df_all_events.empty: # Check if regression was successful and df_all_events is not None/empty
            print(f"\n--- Analyzing Event Populations for LOM {current_lom_name} based on Charge Correlation ---")
            # Calculate residuals (distance from the fitted line)
            df_all_events['predicted_ch2_pC'] = linear_function(df_all_events['ch1_pC'], m_slope, c_intercept)
            df_all_events['residual'] = np.abs(df_all_events['ch2_pC'] - df_all_events['predicted_ch2_pC'])

            # Define a threshold for "well-correlated" vs "outlier"
            residual_threshold_pC = 5.0 # This threshold can be adjusted.

            well_correlated_events = df_all_events[df_all_events['residual'] <= residual_threshold_pC]
            outlier_events = df_all_events[df_all_events['residual'] > residual_threshold_pC]

            print(f"Total events analyzed for populations: {len(df_all_events)}")
            print(f"Well-correlated events (residual <= {residual_threshold_pC} pC): {len(well_correlated_events)}")
            print(f"Outlier events (residual > {residual_threshold_pC} pC): {len(outlier_events)}")

            # Sample and plot waveforms for a few well-correlated events
            if not well_correlated_events.empty:
                sample_well_correlated = well_correlated_events.sample(min(len(well_correlated_events), args.waveform_sample_size), random_state=42).to_dict('records')
                for event in sample_well_correlated:
                    event['event_type'] = 'Well-Correlated'
                plot_sample_waveforms(
                    sample_well_correlated,
                    f"Sample Waveforms (Well-Correlated) - LOM {current_lom_name}",
                    f"{lom_output_prefix}sample_waveforms_well_correlated",
                    output_dirs["waveform_samples"],
                    x_range_mode="peak_focus"
                )
                plot_sample_waveforms(
                    sample_well_correlated,
                    f"Sample Waveforms (Well-Correlated) - LOM {current_lom_name}",
                    f"{lom_output_prefix}sample_waveforms_well_correlated",
                    output_dirs["waveform_samples"],
                    x_range_mode="broad"
                )
            else:
                print("No well-correlated events to sample.")

            # Sample and plot waveforms for a few outlier events
            if not outlier_events.empty:
                sample_outliers = outlier_events.sample(min(len(outlier_events), args.waveform_sample_size), random_state=42).to_dict('records')
                for event in sample_outliers:
                    event['event_type'] = 'Outlier'
                plot_sample_waveforms(
                    sample_outliers,
                    f"Sample Waveforms (Outliers) - LOM {current_lom_name}",
                    f"{lom_output_prefix}sample_waveforms_outliers",
                    output_dirs["waveform_samples"],
                    x_range_mode="peak_focus"
                )
                plot_sample_waveforms(
                    sample_outliers,
                    f"Sample Waveforms (Outliers) - LOM {current_lom_name}",
                    f"{lom_output_prefix}sample_waveforms_outliers",
                    output_dirs["waveform_samples"],
                    x_range_mode="broad"
                )
            else:
                print("No outlier events to sample.")
        else:
            print(f"Skipping event population analysis for {current_lom_name} due to no valid regression or no event data.")

        # --- Coincidence Analysis block for current LOM ---
        print("\n--- Starting Coincidence Analysis ---")

        all_pmt_event_times = {}
        all_pmt_livetimes = {}

        print("Calculating Livetimes and Filtering Events for Coincidence (for all PMTs)...")
        pmt_ids_with_data = sorted(pmt_to_infile_map.keys())

        if not pmt_ids_with_data:
            print(f"No valid PMT data for LOM {current_lom_name}. Skipping coincidence analysis.")
            continue

        for pmt_id in pmt_ids_with_data:
            infile_pmt = pmt_to_infile_map[pmt_id]

            data_all_fpga = read_hdffile(infile_pmt)

            all_raw_fpga_times = np.array(data_all_fpga["data"]["FPGAtime"][()] if "data" in data_all_fpga and "FPGAtime" in data_all_fpga["data"] else [])

            pmt_livetime = 0.0
            if all_raw_fpga_times.size > 1:
                start_t = all_raw_fpga_times[0]
                end_t = all_raw_fpga_times[-1]
                pmt_livetime = (end_t - start_t) / 1e9 # Convert ns to seconds
                if pmt_livetime <= 0: pmt_livetime = 0.0

            all_pmt_livetimes[pmt_id] = pmt_livetime

            # Using a broad threshold for general timestamp collection for livetime calc and coincidence
            filtered_timestamps_broad, _ = get_timestamps(infile_pmt, min_charge_pC=0.0, max_charge_pC=float('inf'))
            all_pmt_event_times[pmt_id] = np.sort(np.array(filtered_timestamps_broad))

            print(f"PMT {pmt_id}: Total Filtered Events (broad threshold) = {len(filtered_timestamps_broad)}, Calculated Livetime = {pmt_livetime:.2f}s")


        lom18_x_labels_order = np.array([14,12,10,8,6,4,2,0,1,3,5,7,9,11,13,15])
        lom18_y_labels_order = np.array([15,13,11,9,7,5,3,1,0,2,4,6,8,10,12,14])

        lom18_x_indices_filtered = [p for p in lom18_x_labels_order if p in pmt_ids_with_data]
        lom18_y_indices_filtered = [p for p in lom18_y_labels_order if p in pmt_ids_with_data]

        if not lom18_x_indices_filtered or not lom18_y_indices_filtered:
            print(f"Not enough PMT data for LOM {current_lom_name} to create LOM18 style coincidence matrix plots. Skipping.")
        else:
            # fixed_cbar_ticks and fixed_cbar_vmax are defined outside the loop
            
            # Loop for Coincidence Matrices
            for c_idx, current_threshold_pC in enumerate(COINCIDENCE_THRESHOLDS_PC): # Iterate over thresholds
                print(f"\n--- Building Coincidence Matrix (Threshold: {current_threshold_pC:.1f} pC, Window: {FIXED_COINCIDENCE_WINDOW_NS} ns) ---")

                current_pmt_event_times_filtered = {}
                for pmt_id in pmt_ids_with_data:
                    infile_pmt = pmt_to_infile_map[pmt_id]
                    filtered_ts_current_th, _ = get_timestamps(infile_pmt, min_charge_pC=current_threshold_pC, max_charge_pC=float('inf'))
                    current_pmt_event_times_filtered[pmt_id] = np.sort(np.array(filtered_ts_current_th))

                max_pmt_id_in_data = max(pmt_ids_with_data) if pmt_ids_with_data else -1

                if max_pmt_id_in_data == -1:
                    print("No valid PMT data found to build coincidence matrix. Skipping.")
                    continue

                coincidence_matrix_counts = np.zeros((max_pmt_id_in_data + 1, max_pmt_id_in_data + 1), dtype=int)
                coincidence_matrix_rates = np.zeros((max_pmt_id_in_data + 1, max_pmt_id_in_data + 1), dtype=float)

                for i_pmt in pmt_ids_with_data:
                    for j_pmt in pmt_ids_with_data:
                        if i_pmt == j_pmt:
                            continue

                        times_i = current_pmt_event_times_filtered.get(i_pmt, np.array([]))
                        times_j = current_pmt_event_times_filtered.get(j_pmt, np.array([]))

                        if times_i.size > 0 and times_j.size > 0:
                            _, count = event_matching(times_i, times_j, window_ns=FIXED_COINCIDENCE_WINDOW_NS)
                        else:
                            count = 0

                        coincidence_matrix_counts[i_pmt, j_pmt] = count

                        livetime_i = all_pmt_livetimes.get(i_pmt, 0.0)
                        livetime_j = all_pmt_livetimes.get(j_pmt, 0.0)

                        common_livetime = min(livetime_i, livetime_j)
                        if common_livetime > 0:
                            coincidence_matrix_rates[i_pmt, j_pmt] = count / common_livetime
                        else:
                            coincidence_matrix_rates[i_pmt, j_pmt] = 0.0

                print("Coincidence Counts Matrix for current settings:")
                print(coincidence_matrix_counts[np.ix_(lom18_y_indices_filtered, lom18_x_indices_filtered)])
                print("\nCoincidence Rates Matrix (Hz) for current settings:")
                print(coincidence_matrix_rates[np.ix_(lom18_y_indices_filtered, lom18_x_indices_filtered)])

                # --- Plotting Combined Coincidence Matrix (Linear Scale) ---
                ax_matrix_sub_linear_combined = fig_matrices_all_loms_linear.add_subplot(gs_matrices_linear[lom_idx, c_idx])
                coincidence_matrix_rates_ordered = coincidence_matrix_rates[np.ix_(lom18_y_indices_filtered, lom18_x_indices_filtered)]

                mesh_linear_combined_current = ax_matrix_sub_linear_combined.pcolormesh(np.arange(len(lom18_x_indices_filtered)),
                                                                                         np.arange(len(lom18_y_indices_filtered)),
                                                                                         coincidence_matrix_rates_ordered,
                                                                                         cmap='Blues', edgecolors='white', linewidth=1.0,
                                                                                         vmin=0, vmax=fixed_cbar_vmax)
                matrices_linear_mappables.append(mesh_linear_combined_current) # Collect mappable for common cbar

                ax_matrix_sub_linear_combined.set_xticks(np.arange(len(lom18_x_indices_filtered)) + 0.5)
                ax_matrix_sub_linear_combined.set_yticks(np.arange(len(lom18_y_indices_filtered)) + 0.5)
                ax_matrix_sub_linear_combined.set_xticklabels(lom18_x_indices_filtered, fontsize=7)
                ax_matrix_sub_linear_combined.set_yticklabels(lom18_y_indices_filtered, fontsize=7)

                ax_matrix_sub_linear_combined.set_xlabel('Ch ID', fontsize=8) # Shorter label
                ax_matrix_sub_linear_combined.set_ylabel('Ch ID', fontsize=8) # Shorter label
                ax_matrix_sub_linear_combined.set_title(f"LOM {current_lom_name}\nTh: {current_threshold_pC:.1f} pC", fontsize=9)

                ax_matrix_sub_linear_combined.invert_yaxis()
                ax_matrix_sub_linear_combined.set_aspect('equal', adjustable='box')

                try: # Add Hemisphere Dividing Lines and Labels for each subplot
                    split_idx_x_for_0 = lom18_x_indices_filtered.index(0)
                    split_idx_y_for_0 = lom18_y_indices_filtered.index(0)

                    ax_matrix_sub_linear_combined.axvline(split_idx_x_for_0 + 0.5, color='black', linestyle='-', linewidth=1.5)
                    ax_matrix_sub_linear_combined.axhline(split_idx_y_for_0 + 0.5, color='black', linestyle='-', linewidth=1.5)
                except ValueError:
                    warnings.warn("PMT 0 not found in filtered axis order. Cannot draw hemisphere lines/labels precisely for current plot.")

                # --- Plotting Combined Coincidence Matrix (Log Scale) ---
                ax_matrix_sub_log_combined = fig_matrices_all_loms_log.add_subplot(gs_matrices_log[lom_idx, c_idx])
                coincidence_matrix_rates_ordered_log = np.where(coincidence_matrix_rates_ordered > 0, coincidence_matrix_rates_ordered, 1e-9)

                mesh_log_combined_current = ax_matrix_sub_log_combined.pcolormesh(np.arange(len(lom18_x_indices_filtered)),
                                                                                   np.arange(len(lom18_y_indices_filtered)),
                                                                                   coincidence_matrix_rates_ordered_log,
                                                                                   cmap='Blues', edgecolors='white', linewidth=1.0,
                                                                                   norm=LogNorm(vmin=1e-3, vmax=fixed_cbar_vmax))
                matrices_log_mappables.append(mesh_log_combined_current) # Collect mappable for common cbar

                ax_matrix_sub_log_combined.set_xticks(np.arange(len(lom18_x_indices_filtered)) + 0.5)
                ax_matrix_sub_log_combined.set_yticks(np.arange(len(lom18_y_indices_filtered)) + 0.5)
                ax_matrix_sub_log_combined.set_xticklabels(lom18_x_indices_filtered, fontsize=7)
                ax_matrix_sub_log_combined.set_yticklabels(lom18_y_indices_filtered, fontsize=7)

                ax_matrix_sub_log_combined.set_xlabel('Ch ID', fontsize=8)
                ax_matrix_sub_log_combined.set_ylabel('Ch ID', fontsize=8)
                ax_matrix_sub_log_combined.set_title(f"LOM {current_lom_name}\nTh: {current_threshold_pC:.1f} pC", fontsize=9)

                ax_matrix_sub_log_combined.invert_yaxis()
                ax_matrix_sub_log_combined.set_aspect('equal', adjustable='box')

                try: # Add Hemisphere Dividing Lines and Labels for each subplot
                    split_idx_x_for_0 = lom18_x_indices_filtered.index(0)
                    split_idx_y_for_0 = lom18_y_indices_filtered.index(0)

                    ax_matrix_sub_log_combined.axvline(split_idx_x_for_0 + 0.5, color='black', linestyle='-', linewidth=1.5)
                    ax_matrix_sub_log_combined.axhline(split_idx_y_for_0 + 0.5, color='black', linestyle='-', linewidth=1.5)
                except ValueError:
                    warnings.warn("PMT 0 not found in filtered axis order. Cannot draw hemisphere lines/labels precisely for current plot.")
                
                # --- Plotting Total Coincidence Rate (Linear Scale) --- (Still separate figures as per previous structure)
                # Keep original loop for rates plots, as they are not explicitly requested to be combined into the matrices figure.
                # If these should also be combined into one figure with many subplots, please specify.
                # For now, these plots are generated per LOM as before.
                fig_rates_total_per_lom, ax_rates_total_per_lom = plt.subplots(figsize=(6, 4))
                total_coincidence_rates_per_pmt = np.sum(coincidence_matrix_rates, axis=1)
                pmts_to_plot_rate_sorted = sorted([p for p in pmt_ids_with_data if p <= max_pmt_id_in_data])
                rates_to_plot_sorted = [total_coincidence_rates_per_pmt[p] for p in pmts_to_plot_rate_sorted]
                ax_rates_total_per_lom.plot(pmts_to_plot_rate_sorted, rates_to_plot_sorted, color='blue', marker='o', linestyle='-', markersize=4, label='Data')
                if len(pmts_to_plot_rate_sorted) > 2:
                    try:
                        popt, pcov = curve_fit(polynomial_function, np.array(pmts_to_plot_rate_sorted), np.array(rates_to_plot_sorted))
                        x_fit_poly = np.linspace(min(pmts_to_plot_rate_sorted), max(pmts_to_plot_rate_sorted), 100)
                        y_fit_poly = polynomial_function(x_fit_poly, *popt)
                        ax_rates_total_per_lom.plot(x_fit_poly, y_fit_poly, color='red', linestyle='--', label='Polynomial Fit')
                    except RuntimeError:
                        warnings.warn(f"Could not fit polynomial curve for LOM {current_lom_name}, Th {current_threshold_pC:.1f} pC.")
                ax_rates_total_per_lom.legend(fontsize=7)
                ax_rates_total_per_lom.set_xlabel('PMT Channel ID',fontsize=8)
                ax_rates_total_per_lom.set_ylabel('Total Rate [1/s]',fontsize=8)
                ax_rates_total_per_lom.set_title(f"Total Coincidence Rate - LOM {current_lom_name}\nTh: {current_threshold_pC:.1f} pC, dT: {FIXED_COINCIDENCE_WINDOW_NS}ns", fontsize=9)
                ax_rates_total_per_lom.tick_params(labelsize=7)
                ax_rates_total_per_lom.grid('xy', linestyle=':', lw=0.5)
                fig_rates_total_per_lom.tight_layout()
                fig_rates_total_per_lom.savefig(os.path.join(output_dirs["coincidence_rates"], f"{lom_output_prefix}total_coincidence_rate_Th_{current_threshold_pC:.1f}pC_dT_{FIXED_COINCIDENCE_WINDOW_NS}ns.pdf"))
                plt.close(fig_rates_total_per_lom)

                # Log Scale version of total coincidence rate
                fig_rates_total_log_per_lom, ax_rates_total_log_per_lom = plt.subplots(figsize=(6, 4))
                valid_rates_indices = np.array(rates_to_plot_sorted) > 0 
                if np.any(valid_rates_indices):
                    ax_rates_total_log_per_lom.plot(np.array(pmts_to_plot_rate_sorted)[valid_rates_indices], np.array(rates_to_plot_sorted)[valid_rates_indices], color='blue', marker='o', linestyle='-', markersize=4, label='Data')
                    ax_rates_total_log_per_lom.set_yscale('log')
                    if np.sum(valid_rates_indices) > 2:
                        try:
                            popt_log, pcov_log = curve_fit(polynomial_function, np.array(pmts_to_plot_rate_sorted)[valid_rates_indices], np.log(np.array(rates_to_plot_sorted)[valid_rates_indices]))
                            x_fit_poly = np.linspace(min(np.array(pmts_to_plot_rate_sorted)[valid_rates_indices]), max(np.array(pmts_to_plot_rate_sorted)[valid_rates_indices]), 100)
                            y_fit_poly_log = np.exp(polynomial_function(x_fit_poly, *popt_log))
                            ax_rates_total_log_per_lom.plot(x_fit_poly, y_fit_poly_log, color='red', linestyle='--', label='Log-Polynomial Fit')
                        except RuntimeError:
                            warnings.warn(f"Could not fit log-polynomial curve for LOM {current_lom_name}, Th {current_threshold_pC:.1f} pC.")
                    ax_rates_total_log_per_lom.legend(fontsize=7)
                else:
                    ax_rates_total_log_per_lom.text(0.5, 0.5, "No positive rates to plot", horizontalalignment='center', verticalalignment='center', transform=ax_rates_total_log_per_lom.transAxes, fontsize=10, color='gray')
                
                ax_rates_total_log_per_lom.set_xlabel('PMT Channel ID',fontsize=8)
                ax_rates_total_log_per_lom.set_ylabel('Total Rate [1/s] (Log Scale)',fontsize=8)
                ax_rates_total_log_per_lom.set_title(f"Total Coincidence Rate (Log Scale) - LOM {current_lom_name}\nTh: {current_threshold_pC:.1f} pC, dT: {FIXED_COINCIDENCE_WINDOW_NS}ns", fontsize=9)
                ax_rates_total_log_per_lom.tick_params(labelsize=7)
                ax_rates_total_log_per_lom.grid('xy', linestyle=':', lw=0.5)
                fig_rates_total_log_per_lom.tight_layout()
                fig_rates_total_log_per_lom.savefig(os.path.join(output_dirs["coincidence_rates_log"], f"{lom_output_prefix}total_coincidence_rate_log_Th_{current_threshold_pC:.1f}pC_dT_{FIXED_COINCIDENCE_WINDOW_NS}ns.pdf"))
                plt.close(fig_rates_total_log_per_lom)

    # --- Finalize and save all combined plots ---
    # Charge Correlation: Add common colorbars
    if linear_corr_mappables:
        # Create a single colorbar for the linear plots spanning the height of the linear row
        cbar_ax_linear_corr = fig_corr_combined.add_subplot(gs_corr[0, -1]) # Last column, top row
        fig_corr_combined.colorbar(linear_corr_mappables[0], cax=cbar_ax_linear_corr, label='Number of Events (Linear Scale)')
    if log_corr_mappables:
        # Create a single colorbar for the log plots spanning the height of the log row
        cbar_ax_log_corr = fig_corr_combined.add_subplot(gs_corr[1, -1]) # Last column, bottom row
        fig_corr_combined.colorbar(log_corr_mappables[0], cax=cbar_ax_log_corr, label='Number of Events (Log Scale)', format=LogNorm())
    
    # Adjust tight_layout to account for the new GridSpec
    fig_corr_combined.tight_layout()
    corr_combined_filename = os.path.join(output_dirs["charge_correlations"], f"All_LOMs_charge_correlation_combined_landscape.pdf")
    fig_corr_combined.savefig(corr_combined_filename)
    plt.close(fig_corr_combined)
    print(f'Saved combined charge correlation for all LOMs to: {corr_combined_filename}')


    # Coincidence Matrices (Linear Scale): Add common colorbar
    if matrices_linear_mappables:
        # Create a single colorbar for all linear matrices
        cbar_ax_linear_matrices = fig_matrices_all_loms_linear.add_subplot(gs_matrices_linear[:, -1]) # Last column, all rows
        fig_matrices_all_loms_linear.colorbar(matrices_linear_mappables[0], cax=cbar_ax_linear_matrices, 
                                              label='Coincidence Rate [Hz]', ticks=fixed_cbar_ticks)
    fig_matrices_all_loms_linear.tight_layout()
    matrices_linear_filename = os.path.join(output_dirs["coincidence_matrices"], f"All_LOMs_coincidence_matrices_linear_dT_{FIXED_COINCIDENCE_WINDOW_NS}ns_landscape.pdf")
    fig_matrices_all_loms_linear.savefig(matrices_linear_filename)
    plt.close(fig_matrices_all_loms_linear)
    print(f'Saved combined linear coincidence matrices for all LOMs to: {matrices_linear_filename}')

    # Coincidence Matrices (Log Scale): Add common colorbar
    if matrices_log_mappables:
        # Create a single colorbar for all log matrices
        cbar_ax_log_matrices = fig_matrices_all_loms_log.add_subplot(gs_matrices_log[:, -1]) # Last column, all rows
        fig_matrices_all_loms_log.colorbar(matrices_log_mappables[0], cax=cbar_ax_log_matrices, 
                                           label='Coincidence Rate [Hz] (Log Scale)', format='%.2e')
    fig_matrices_all_loms_log.tight_layout()
    matrices_log_filename = os.path.join(output_dirs["coincidence_matrices_log"], f"All_LOMs_coincidence_matrices_log_dT_{FIXED_COINCIDENCE_WINDOW_NS}ns_landscape.pdf")
    fig_matrices_all_loms_log.savefig(matrices_log_filename)
    plt.close(fig_matrices_all_loms_log)
    print(f'Saved combined log coincidence matrices for all LOMs to: {matrices_log_filename}')


    print("\n========================================================")
    print("=== All LOMs Processed. Analysis Complete. ===")
    print("========================================================")