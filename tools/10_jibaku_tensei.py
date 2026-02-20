# 10_jibaku_tensei.py
#
# This script is a refactored version focusing on Nobu's LOM-18 style coincidence analysis,
# waveform preprocessing, and FFT analysis, with new features for charge correlation and gain calculation.
#
# Features:
# - Reads HDF files (Muon or SPE data) from a hierarchical LOM folder structure.
# - Generates PMT vs Coincidence Rate plot per LOM.
# - Generates Coincidence Matrix (heatmap) for pairwise coincidences with LOM18 style per LOM.
# - Performs Fast Fourier Transform (FFT) analysis on waveforms of 100 pC peak events per LOM.
# - Generates comparison plots (Time-Domain and Frequency-Domain) across all 16 PMTs per LOM.
# - Implements LOM18 style ordering, lines, and labels for Coincidence Matrix axes.
# - Calculates livetime based on FPGA timestamps.
# - Allows specifying an output directory for all saved plots.
# - Waveform preprocessing (Pedestal subtraction, Ch2 polarity inversion) applied.
# - Coincidence matrix generated for multiple specified thresholds and a FIXED 100 ns time window.
# - Organizes output plots into subfolders, including LOM-specific subfolders.
# - Plots Muon Charge Distribution for Channel 2 in PE (linear scale) per LOM.
# - NEW: Plots Ch1 vs Ch2 Charge Correlation for events, performs linear regression (Q2 = mQ1 + c),
#        and calculates the 'm' (slope) as the gain factor for Ch2 relative to Ch1 (where Ch1 has 0.8 pC/PE).
# - MODIFIED: Coincidence Matrix color scale fixed and log scale version added.
# - MODIFIED: Coincidence Rate plots include a fitting curve.
# - MODIFIED: Charge Correlation plot axis ranges adjusted.
#
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
    q_ch1_pC_all = np.array([ s * (conversion_ch1 * 1e-6 * (1/60e6) * 1e12) for s in q_ch1])

    q_ch2 = data["data"]["charge_ch2"][()] if "charge_ch2" in data["data"] else np.array([])
    conversion_ch2 = data["metadata"]["conversion_ch2"][()] if "conversion_ch2" in data["metadata"] else 1.0
    q_ch2_pC_all = np.array([ s * (conversion_ch2 * 1e-6 * (1/60e6) * 1e12) for s in q_ch2])

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
            charges_pC = raw_charges_ch2_adc * (conversion_ch2 * 1e-6 * (1/60e6) * 1e12) # Re-use consistent conversion
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

    bins = np.linspace(0, 10000, 201) # 200 bins for 0-10000 PE

    ax.hist(all_ch2_charges_pe, bins=bins, color='skyblue', edgecolor='blue', alpha=0.7)

    ax.set_xlabel('Charge (PE)', fontsize=12)
    ax.set_ylabel('Counts (Linear Scale)', fontsize=12)
    ax.set_title(f'Muon Charge Distribution: Channel 2 (All PMTs Combined) - {lom_output_prefix.strip("_")}', fontsize=14)
    
    ax.set_xlim([0, 10000]) # X-axis range 0-10000 PE
    ax.set_yscale('linear') # Ensure linear scale as requested

    ax.tick_params(labelsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)

    output_filename = f"{lom_output_prefix}muon_charge_distribution_ch2_linear_scale.pdf"
    plt.savefig(os.path.join(output_dir_dist, output_filename))
    plt.close(fig)
    print(f"Saved Muon Charge Distribution (Ch2) plot to: {os.path.join(output_dir_dist, output_filename)}")


def plot_charge_correlation_and_gain(pmt_to_infile_map, assumed_ch1_pC_per_PE, output_dir, lom_output_prefix=""):
    """
    Plots Ch1 vs Ch2 charge correlation for all events across all PMTs within a LOM,
    performs linear regression (Q2 = mQ1 + c), and calculates the gain factor 'm'
    for Ch2 relative to Ch1.
    
    Args:
        pmt_to_infile_map (dict): Map of PMT ID to HDF file path for the current LOM.
        assumed_ch1_pC_per_PE (float): The assumed pC per PE for Channel 1 (e.g., 0.8 pC/PE).
        output_dir (str): Directory to save the plot.
        lom_output_prefix (str): Prefix for the output filename (e.g., "LOM16-01_").
    """
    all_ch1_charges_pC = []
    all_ch2_charges_pC = []

    print(f"\n--- Generating Ch1 vs Ch2 Charge Correlation for {lom_output_prefix.strip('_')} ---")

    for pmt_id, infile_path in pmt_to_infile_map.items():
        data = read_hdffile(infile_path)
        if not data or "data" not in data or "metadata" not in data:
            warnings.warn(f"Missing 'data' or 'metadata' group in {infile_path}. Skipping charge correlation for PMT {pmt_id}.")
            continue

        try:
            raw_charges_ch1_adc = data["data"]["charge_ch1"][()]
            conversion_ch1 = data["metadata"]["conversion_ch1"][()]
            raw_charges_ch2_adc = data["data"]["charge_ch2"][()]
            conversion_ch2 = data["metadata"]["conversion_ch2"][()]
        except KeyError as e:
            warnings.warn(f"Missing charge or conversion data for PMT {pmt_id} in {infile_path}: {e}. Skipping for correlation.")
            continue
        
        conversion_factor_pC = (1e-6 * (1/60e6) * 1e12) # consistent conversion factor

        charges_ch1_pC = raw_charges_ch1_adc * (conversion_ch1 * conversion_factor_pC)
        charges_ch2_pC = raw_charges_ch2_adc * (conversion_ch2 * conversion_factor_pC)

        # Filter out NaN/inf and align lengths
        min_len = min(len(charges_ch1_pC), len(charges_ch2_pC))
        all_ch1_charges_pC.extend(charges_ch1_pC[:min_len])
        all_ch2_charges_pC.extend(charges_ch2_pC[:min_len])

    if not all_ch1_charges_pC or not all_ch2_charges_pC:
        print(f"No valid Ch1/Ch2 charge data found for correlation plot for {lom_output_prefix.strip('_')}.")
        return

    all_ch1_charges_pC = np.array(all_ch1_charges_pC)
    all_ch2_charges_pC = np.array(all_ch2_charges_pC)

    # Filter out events with zero or very low charge that might skew regression
    # Or filter for a certain range relevant for "muon-like" events if desired
    # Let's consider events where both charges are above a small threshold (e.g., 1 pC)
    valid_indices = (all_ch1_charges_pC > 1) & (all_ch2_charges_pC > 1)
    x_data = all_ch1_charges_pC[valid_indices]
    y_data = all_ch2_charges_pC[valid_indices]

    if len(x_data) < 2: # Need at least 2 points for regression
        print(f"Not enough valid charge data points for linear regression for {lom_output_prefix.strip('_')}. Skipping correlation plot.")
        return

    try:
        # Perform linear regression: Q2 = m*Q1 + c
        params, cov = curve_fit(linear_function, x_data, y_data)
        m_slope, c_intercept = params
        
        # Calculate R-squared (coefficient of determination)
        residuals = y_data - linear_function(x_data, m_slope, c_intercept)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        print(f"For {lom_output_prefix.strip('_')}: Linear Regression (Q2 = m*Q1 + c)")
        print(f"Slope (m) = {m_slope:.4f}")
        print(f"Intercept (c) = {c_intercept:.4f} pC")
        print(f"R-squared = {r_squared:.4f}")

    except RuntimeError:
        print(f"Could not find optimal parameters for linear regression for {lom_output_prefix.strip('_')}.")
        m_slope = np.nan
        c_intercept = np.nan
        r_squared = np.nan

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(x_data, y_data, alpha=0.3, s=5, label=f'Data ({len(x_data)} points)') # Use filtered data
    
    # Plot the regression line
    if not np.isnan(m_slope):
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = linear_function(x_fit, m_slope, c_intercept)
        ax.plot(x_fit, y_fit, color='red', linestyle='--', linewidth=2,
                label=f'Fit: Q2 = {m_slope:.2f}*Q1 + {c_intercept:.2f} pC\n($R^2={r_squared:.2f}$)')

    ax.set_xlabel('Charge Ch1 (pC)', fontsize=12)
    ax.set_ylabel('Charge Ch2 (pC)', fontsize=12)
    ax.set_title(f'Ch1 vs Ch2 Charge Correlation - {lom_output_prefix.strip("_")}', fontsize=14)
    ax.set_xlim(0, 300) # MODIFIED: Set X-axis range to (0 - 500) pC
    ax.set_ylim(0, 60)  # MODIFIED: Set Y-axis range to (0 - 50) pC
    ax.tick_params(labelsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=10)

    output_filename = f"{lom_output_prefix}charge_correlation_ch1_ch2.pdf"
    plt.savefig(os.path.join(output_dir, output_filename))
    plt.close(fig)
    print(f"Saved Charge Correlation plot to: {os.path.join(output_dir, output_filename)}")

# --- End: Plotting Functions ---


if __name__ == "__main__":
    # Configure command-line argument parser
    parser = argparse.ArgumentParser(description="Streamlined Muon Analysis Script: Focuses on coincidence and FFT analysis.")
    parser.add_argument("lom_data_dirs", nargs='+', help="Input LOM data directories (e.g., 'path/to/LOM16-01/'). Each directory should contain 16 PMT HDF files.", type=str)
    parser.add_argument("--output_dir", "-o", help="Output directory for plots (default: ./plots_output)", type=str, default='./plots_output')
    parser.add_argument("--assumed_gain", type=float, default=5e6, help="Assumed PMT gain for PE conversion (electrons/PE). Default is 5e6.")
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
        "charge_correlations": os.path.join(output_base_dir, "charge_correlations")
    }
    for d in output_dirs.values():
        os.makedirs(d, exist_ok=True)
    print(f"Output plots will be saved to: {output_base_dir} and its subfolders.")

    # --- Start: Main loop for LOM-by-LOM processing ---
    # Sort LOM directories for consistent processing order
    sorted_lom_data_dirs = sorted(args.lom_data_dirs)

    # List to hold all figures so they can be shown at the end (if plt.show() is uncommented)
    all_figures = []

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
        # Assuming HDF files are directly inside the LOM folder (e.g., LOM_XX/data-muon-run909.00.hdf)
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


        # --- NEW: Charge Correlation Plot (Ch1 vs Ch2) for the current LOM ---
        plot_charge_correlation_and_gain(
            pmt_to_infile_map,
            ASSUMED_CH1_PC_PER_PE,
            output_dirs["charge_correlations"],
            lom_output_prefix
        )

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

        # Now, fix the time window to 100 ns as requested
        fixed_coincidence_windows_for_plots = [FIXED_COINCIDENCE_WINDOW_NS]

        n_rows_combined = len(fixed_coincidence_windows_for_plots) # Now always 1
        n_cols_combined = len(COINCIDENCE_THRESHOLDS_PC) # Still 4 columns for thresholds

        # --- Linear Scale Coincidence Plots ---
        fig_matrices, axes_matrices = plt.subplots(n_rows_combined, n_cols_combined, figsize=(5.5 * n_cols_combined, 4.5 * n_rows_combined), squeeze=False)
        fig_matrices.suptitle(f"Coincidence Rate Matrices (LOM: {current_lom_name}, Fixed ΔT: {FIXED_COINCIDENCE_WINDOW_NS} ns)", fontsize=16, y=1.02)

        fig_rates_total, axes_rates_total = plt.subplots(n_rows_combined, n_cols_combined, figsize=(5.5 * n_cols_combined, 4.5 * n_rows_combined), squeeze=False)
        fig_rates_total.suptitle(f"Total Coincidence Rates per PMT (LOM: {current_lom_name}, Fixed ΔT: {FIXED_COINCIDENCE_WINDOW_NS} ns)", fontsize=16, y=1.02)

        # --- Log Scale Coincidence Plots ---
        fig_matrices_log, axes_matrices_log = plt.subplots(n_rows_combined, n_cols_combined, figsize=(5.5 * n_cols_combined, 4.5 * n_rows_combined), squeeze=False)
        fig_matrices_log.suptitle(f"Coincidence Rate Matrices (Log Scale) (LOM: {current_lom_name}, Fixed ΔT: {FIXED_COINCIDENCE_WINDOW_NS} ns)", fontsize=16, y=1.02)

        fig_rates_total_log, axes_rates_total_log = plt.subplots(n_rows_combined, n_cols_combined, figsize=(5.5 * n_cols_combined, 4.5 * n_rows_combined), squeeze=False)
        fig_rates_total_log.suptitle(f"Total Coincidence Rates per PMT (Log Scale) (LOM: {current_lom_name}, Fixed ΔT: {FIXED_COINCIDENCE_WINDOW_NS} ns)", fontsize=16, y=1.02)


        lom18_x_labels_order = np.array([14,12,10,8,6,4,2,0,1,3,5,7,9,11,13,15])
        lom18_y_labels_order = np.array([15,13,11,9,7,5,3,1,0,2,4,6,8,10,12,14])

        lom18_x_indices_filtered = [p for p in lom18_x_labels_order if p in pmt_ids_with_data]
        lom18_y_indices_filtered = [p for p in lom18_y_labels_order if p in pmt_ids_with_data]

        if not lom18_x_indices_filtered or not lom18_y_indices_filtered:
            print(f"Not enough PMT data for LOM {current_lom_name} to create LOM18 style coincidence matrix plots. Skipping.")
        else:
            # Define a single colorbar for the matrices to avoid clutter
            cbar_axes_linear = None
            cbar_axes_log = None

            # Define fixed colorbar ticks for linear scale
            fixed_cbar_ticks = np.array([0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00])
            fixed_cbar_vmax = fixed_cbar_ticks[-1] # Max value for colorbar

            for r_idx, current_window_ns in enumerate(fixed_coincidence_windows_for_plots): # Will only loop once now
                for c_idx, current_threshold_pC in enumerate(COINCIDENCE_THRESHOLDS_PC): # Iterate over thresholds
                    print(f"\n--- Building Coincidence Matrix (Threshold: {current_threshold_pC:.1f} pC, Window: {current_window_ns} ns) ---")

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
                                _, count = event_matching(times_i, times_j, window_ns=current_window_ns)
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

                    # --- Plotting Coincidence Matrix (Linear Scale) ---
                    ax_matrix_sub = axes_matrices[r_idx, c_idx]
                    coincidence_matrix_rates_ordered = coincidence_matrix_rates[np.ix_(lom18_y_indices_filtered, lom18_x_indices_filtered)]

                    mesh_linear = ax_matrix_sub.pcolormesh(np.arange(len(lom18_x_indices_filtered)),
                                                     np.arange(len(lom18_y_indices_filtered)),
                                                     coincidence_matrix_rates_ordered,
                                                     cmap='Blues', edgecolors='white', linewidth=1.0,
                                                     vmin=0, vmax=fixed_cbar_vmax) # MODIFIED: Fixed vmax

                    if cbar_axes_linear is None: # Add colorbar only once for the whole figure
                        cbar_ax_linear = fig_matrices.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
                        cbar_linear = fig_matrices.colorbar(mesh_linear, cax=cbar_ax_linear, orientation='vertical', ticks=fixed_cbar_ticks) # MODIFIED: Fixed ticks
                        cbar_linear.set_label('Coincidence Rate [Hz]', fontsize=10)
                        cbar_axes_linear = cbar_ax_linear # Store for subsequent plots

                    ax_matrix_sub.set_xticks(np.arange(len(lom18_x_indices_filtered)) + 0.5)
                    ax_matrix_sub.set_yticks(np.arange(len(lom18_y_indices_filtered)) + 0.5)
                    ax_matrix_sub.set_xticklabels(lom18_x_indices_filtered, fontsize=7)
                    ax_matrix_sub.set_yticklabels(lom18_y_indices_filtered, fontsize=7)

                    ax_matrix_sub.set_xlabel('Channel ID', fontsize=8)
                    ax_matrix_sub.set_ylabel('Channel ID', fontsize=8)
                    ax_matrix_sub.set_title(f"Th: {current_threshold_pC:.1f} pC", fontsize=9)

                    ax_matrix_sub.invert_yaxis()
                    ax_matrix_sub.set_aspect('equal', adjustable='box')

                    try: # Add Hemisphere Dividing Lines and Labels for each subplot
                        split_idx_x_for_0 = lom18_x_indices_filtered.index(0)
                        split_idx_y_for_0 = lom18_y_indices_filtered.index(0)

                        ax_matrix_sub.axvline(split_idx_x_for_0 + 0.5, color='black', linestyle='-', linewidth=2)
                        ax_matrix_sub.axhline(split_idx_y_for_0 + 0.5, color='black', linestyle='-', linewidth=2)

                        x_mid_upper = (split_idx_x_for_0 + 0.5) / 2
                        x_mid_lower = ( (split_idx_x_for_0 + 0.5) + len(lom18_x_indices_filtered) ) / 2
                        ax_matrix_sub.text(x_mid_upper / len(lom18_x_indices_filtered), -0.15, '← Upper', color='black', fontsize=6.5, ha='center', va='top', transform=ax_matrix_sub.transAxes)
                        ax_matrix_sub.text(x_mid_lower / len(lom18_x_indices_filtered), -0.15, 'Lower →', color='black', fontsize=6.5, ha='center', va='top', transform=ax_matrix_sub.transAxes)

                        y_mid_upper = ( (split_idx_y_for_0 + 0.5) + len(lom18_y_indices_filtered) ) / 2
                        y_mid_lower = (split_idx_y_for_0 + 0.5) / 2
                        ax_matrix_sub.text(-0.15, y_mid_upper / len(lom18_y_indices_filtered), '← Upper', color='black', fontsize=6.5, ha='right', va='center', rotation=90, transform=ax_matrix_sub.transAxes)
                        ax_matrix_sub.text(-0.15, y_mid_lower / len(lom18_y_indices_filtered), 'Lower →', color='black', fontsize=6.5, ha='right', va='center', rotation=90, transform=ax_matrix_sub.transAxes)

                    except ValueError:
                        warnings.warn("PMT 0 not found in filtered axis order. Cannot draw hemisphere lines/labels precisely for current plot.")

                    # --- Plotting Total Coincidence Rate (Linear Scale) ---
                    ax_rate_sub = axes_rates_total[r_idx, c_idx]
                    total_coincidence_rates_per_pmt = np.sum(coincidence_matrix_rates, axis=1)

                    pmts_to_plot_rate_sorted = sorted([p for p in pmt_ids_with_data if p <= max_pmt_id_in_data])
                    rates_to_plot_sorted = [total_coincidence_rates_per_pmt[p] for p in pmts_to_plot_rate_sorted]

                    ax_rate_sub.plot(pmts_to_plot_rate_sorted, rates_to_plot_sorted, color='blue', marker='o', linestyle='-', markersize=4, label='Data')
                    
                    # MODIFIED: Add fitting curve for Coincidence Rate plots
                    if len(pmts_to_plot_rate_sorted) > 2: # Need at least 3 points for 2nd degree polynomial
                        try:
                            # Fit a 2nd degree polynomial
                            popt, pcov = curve_fit(polynomial_function, np.array(pmts_to_plot_rate_sorted), np.array(rates_to_plot_sorted))
                            x_fit_poly = np.linspace(min(pmts_to_plot_rate_sorted), max(pmts_to_plot_rate_sorted), 100)
                            y_fit_poly = polynomial_function(x_fit_poly, *popt)
                            ax_rate_sub.plot(x_fit_poly, y_fit_poly, color='red', linestyle='--', label='Polynomial Fit')
                        except RuntimeError:
                            warnings.warn(f"Could not fit polynomial curve for LOM {current_lom_name}, Th {current_threshold_pC:.1f} pC.")
                    ax_rate_sub.legend(fontsize=7)


                    ax_rate_sub.set_xlabel('PMT Channel ID',fontsize=8)
                    ax_rate_sub.set_ylabel('Total Rate [1/s]',fontsize=8)
                    ax_rate_sub.set_title(f"Th: {current_threshold_pC:.1f} pC", fontsize=9)
                    ax_rate_sub.tick_params(labelsize=7)
                    ax_rate_sub.grid('xy', linestyle=':', lw=0.5)


                    # --- Plotting Coincidence Matrix (Log Scale) ---
                    ax_matrix_sub_log = axes_matrices_log[r_idx, c_idx]
                    # Replace 0s with a very small number for log scale to avoid errors
                    coincidence_matrix_rates_ordered_log = np.where(coincidence_matrix_rates_ordered > 0, coincidence_matrix_rates_ordered, 1e-9) # Small non-zero value

                    mesh_log = ax_matrix_sub_log.pcolormesh(np.arange(len(lom18_x_indices_filtered)),
                                                     np.arange(len(lom18_y_indices_filtered)),
                                                     coincidence_matrix_rates_ordered_log,
                                                     cmap='Blues', edgecolors='white', linewidth=1.0,
                                                     norm=LogNorm(vmin=1e-3, vmax=fixed_cbar_vmax)) # MODIFIED: LogNorm

                    if cbar_axes_log is None: # Add colorbar only once for the whole figure
                        cbar_ax_log = fig_matrices_log.add_axes([0.92, 0.15, 0.02, 0.7])
                        cbar_log = fig_matrices_log.colorbar(mesh_log, cax=cbar_ax_log, orientation='vertical', format='%.2e') # Log format
                        cbar_log.set_label('Coincidence Rate [Hz] (Log Scale)', fontsize=10)
                        cbar_axes_log = cbar_ax_log # Store for subsequent plots

                    ax_matrix_sub_log.set_xticks(np.arange(len(lom18_x_indices_filtered)) + 0.5)
                    ax_matrix_sub_log.set_yticks(np.arange(len(lom18_y_indices_filtered)) + 0.5)
                    ax_matrix_sub_log.set_xticklabels(lom18_x_indices_filtered, fontsize=7)
                    ax_matrix_sub_log.set_yticklabels(lom18_y_indices_filtered, fontsize=7)

                    ax_matrix_sub_log.set_xlabel('Channel ID', fontsize=8)
                    ax_matrix_sub_log.set_ylabel('Channel ID', fontsize=8)
                    ax_matrix_sub_log.set_title(f"Th: {current_threshold_pC:.1f} pC", fontsize=9)

                    ax_matrix_sub_log.invert_yaxis()
                    ax_matrix_sub_log.set_aspect('equal', adjustable='box')

                    try: # Add Hemisphere Dividing Lines and Labels for each subplot
                        split_idx_x_for_0 = lom18_x_indices_filtered.index(0)
                        split_idx_y_for_0 = lom18_y_indices_filtered.index(0)

                        ax_matrix_sub_log.axvline(split_idx_x_for_0 + 0.5, color='black', linestyle='-', linewidth=2)
                        ax_matrix_sub_log.axhline(split_idx_y_for_0 + 0.5, color='black', linestyle='-', linewidth=2)

                        x_mid_upper = (split_idx_x_for_0 + 0.5) / 2
                        x_mid_lower = ( (split_idx_x_for_0 + 0.5) + len(lom18_x_indices_filtered) ) / 2
                        ax_matrix_sub_log.text(x_mid_upper / len(lom18_x_indices_filtered), -0.15, '← Upper', color='black', fontsize=6.5, ha='center', va='top', transform=ax_matrix_sub_log.transAxes)
                        ax_matrix_sub_log.text(x_mid_lower / len(lom18_x_indices_filtered), -0.15, 'Lower →', color='black', fontsize=6.5, ha='center', va='top', transform=ax_matrix_sub_log.transAxes)

                        y_mid_upper = ( (split_idx_y_for_0 + 0.5) + len(lom18_y_indices_filtered) ) / 2
                        y_mid_lower = (split_idx_y_for_0 + 0.5) / 2
                        ax_matrix_sub_log.text(-0.15, y_mid_upper / len(lom18_y_indices_filtered), '← Upper', color='black', fontsize=6.5, ha='right', va='center', rotation=90, transform=ax_matrix_sub_log.transAxes)
                        ax_matrix_sub_log.text(-0.15, y_mid_lower / len(lom18_y_indices_filtered), 'Lower →', color='black', fontsize=6.5, ha='right', va='center', rotation=90, transform=ax_matrix_sub_log.transAxes)

                    except ValueError:
                        warnings.warn("PMT 0 not found in filtered axis order. Cannot draw hemisphere lines/labels precisely for current plot.")

                    # --- Plotting Total Coincidence Rate (Log Scale) ---
                    ax_rate_sub_log = axes_rates_total_log[r_idx, c_idx]
                    # Filter out zero rates for log scale
                    rates_to_plot_sorted_log = np.array(rates_to_plot_sorted)
                    valid_rates_indices = rates_to_plot_sorted_log > 0
                    
                    if np.any(valid_rates_indices):
                        ax_rate_sub_log.plot(np.array(pmts_to_plot_rate_sorted)[valid_rates_indices], rates_to_plot_sorted_log[valid_rates_indices], color='blue', marker='o', linestyle='-', markersize=4, label='Data')
                        ax_rate_sub_log.set_yscale('log') # MODIFIED: Set Y-axis to log scale

                        # MODIFIED: Add fitting curve for Coincidence Rate plots (Log scale)
                        if np.sum(valid_rates_indices) > 2: # Need at least 3 points for 2nd degree polynomial
                            try:
                                # Fit a 2nd degree polynomial to log of rates
                                # Note: Fitting log(y) vs x is equivalent to fitting y = exp(poly(x))
                                popt_log, pcov_log = curve_fit(polynomial_function, np.array(pmts_to_plot_rate_sorted)[valid_rates_indices], np.log(rates_to_plot_sorted_log[valid_rates_indices]))
                                y_fit_poly_log = np.exp(polynomial_function(x_fit_poly, *popt_log)) # Convert back to linear for plotting
                                ax_rate_sub_log.plot(x_fit_poly, y_fit_poly_log, color='red', linestyle='--', label='Log-Polynomial Fit')
                            except RuntimeError:
                                warnings.warn(f"Could not fit log-polynomial curve for LOM {current_lom_name}, Th {current_threshold_pC:.1f} pC.")
                        ax_rate_sub_log.legend(fontsize=7)
                    else:
                        ax_rate_sub_log.text(0.5, 0.5, "No positive rates to plot", horizontalalignment='center', verticalalignment='center', transform=ax_rate_sub_log.transAxes, fontsize=10, color='gray')


                    ax_rate_sub_log.set_xlabel('PMT Channel ID',fontsize=8)
                    ax_rate_sub_log.set_ylabel('Total Rate [1/s] (Log Scale)',fontsize=8)
                    ax_rate_sub_log.set_title(f"Th: {current_threshold_pC:.1f} pC", fontsize=9)
                    ax_rate_sub_log.tick_params(labelsize=7)
                    ax_rate_sub_log.grid('xy', linestyle=':', lw=0.5)


            # Finalize and save combined plots for the current LOM
            fig_matrices.tight_layout(rect=[0, 0.03, 0.9, 0.98], w_pad=0.5, h_pad=0.5)
            matrix_plot_filename = os.path.join(output_dirs["coincidence_matrices"], f"{lom_output_prefix}combined_coincidence_matrices_dT_{FIXED_COINCIDENCE_WINDOW_NS}ns.pdf")
            fig_matrices.savefig(matrix_plot_filename)
            plt.close(fig_matrices)
            print(f'Saved combined coincidence matrices (linear) to: {matrix_plot_filename}')

            fig_rates_total.tight_layout(rect=[0, 0.03, 1, 0.98], w_pad=0.5, h_pad=0.5)
            rate_plot_filename = os.path.join(output_dirs["coincidence_rates"], f"{lom_output_prefix}combined_total_coincidence_rates_dT_{FIXED_COINCIDENCE_WINDOW_NS}ns.pdf")
            fig_rates_total.savefig(rate_plot_filename)
            plt.close(fig_rates_total)
            print(f'Saved combined total coincidence rates (linear) to: {rate_plot_filename}')

            # Save Log Scale plots
            fig_matrices_log.tight_layout(rect=[0, 0.03, 0.9, 0.98], w_pad=0.5, h_pad=0.5)
            matrix_plot_filename_log = os.path.join(output_dirs["coincidence_matrices_log"], f"{lom_output_prefix}combined_coincidence_matrices_log_dT_{FIXED_COINCIDENCE_WINDOW_NS}ns.pdf")
            fig_matrices_log.savefig(matrix_plot_filename_log)
            plt.close(fig_matrices_log)
            print(f'Saved combined coincidence matrices (log) to: {matrix_plot_filename_log}')

            fig_rates_total_log.tight_layout(rect=[0, 0.03, 1, 0.98], w_pad=0.5, h_pad=0.5)
            rate_plot_filename_log = os.path.join(output_dirs["coincidence_rates_log"], f"{lom_output_prefix}combined_total_coincidence_rates_log_dT_{FIXED_COINCIDENCE_WINDOW_NS}ns.pdf")
            fig_rates_total_log.savefig(rate_plot_filename_log)
            plt.close(fig_rates_total_log)
            print(f'Saved combined total coincidence rates (log) to: {rate_plot_filename_log}')


        # --- End Coincidence Analysis for current LOM ---

        # --- Start: FFT Analysis of 100 pC Peak Waveforms and Comparison Plots ---
        print("\n--- Starting FFT Analysis of 100 pC Peak Waveforms ---")
        # Define the peak charge range for waveform analysis
        min_charge_peak_pC = 80.0
        max_charge_peak_pC = 120.0

        pmt_ch1_wf_data_for_comparison = {}
        pmt_ch1_fft_data_for_comparison = {}
        pmt_ch2_wf_data_for_comparison = {}
        pmt_ch2_fft_data_for_comparison = {}

        for pmt_id in pmt_ids_with_data:
            infile_pmt = pmt_to_infile_map[pmt_id]

            timestamps_in_peak, event_indices_in_peak = get_timestamps(
                infile_pmt,
                min_charge_pC=min_charge_peak_pC,
                max_charge_pC=max_charge_peak_pC
            )

            if not timestamps_in_peak:
                print(f"PMT {pmt_id}: No events found in {min_charge_peak_pC}-{max_charge_peak_pC} pC range. Skipping FFT and comparison for this PMT.")
                pmt_ch1_wf_data_for_comparison[pmt_id] = None
                pmt_ch1_fft_data_for_comparison[pmt_id] = None
                pmt_ch2_wf_data_for_comparison[pmt_id] = None
                pmt_ch2_fft_data_for_comparison[pmt_id] = None
                continue

            # Pick the first event in the peak range as representative
            representative_event_idx = event_indices_in_peak[0]
            representative_timestamp = timestamps_in_peak[0]

            rep_charges_ch1, rep_charges_ch2 = get_charges_of_these_events(infile_pmt, [representative_event_idx])
            rep_charge_ch1_pC = rep_charges_ch1[0] if rep_charges_ch1 else np.nan
            rep_charge_ch2_pC = rep_charges_ch2[0] if rep_charges_ch2 else np.nan

            print(f"PMT {pmt_id}: Analyzing representative Event {representative_event_idx} (Time: {representative_timestamp} ns, Ch1 Charge: {rep_charge_ch1_pC:.2f} pC, Ch2 Charge: {rep_charge_ch2_pC:.2f} pC)")

            x_wf, wf1_data, wf2_data = get_waveform_at_this_timestamp(infile_pmt, representative_timestamp)

            dt_ns = x_wf[1] - x_wf[0] if len(x_wf) > 1 else (1e9 / 60e6) # Time per sample in ns
            dt_s = dt_ns * 1e-9 # Time per sample in seconds
            Fs_Hz = 1.0 / dt_s # Sampling Frequency in Hz

            if len(wf1_data) > 0:
                N_ch1 = len(wf1_data)
                fft_result_ch1 = scipy.fft.fft(wf1_data)
                frequencies_ch1 = scipy.fft.fftfreq(N_ch1, d=dt_s)
                magnitude_spectrum_ch1 = np.abs(fft_result_ch1[:N_ch1 // 2])
                pmt_ch1_wf_data_for_comparison[pmt_id] = {
                    'x_wf': x_wf, 'wf_data': wf1_data, 'ch_label': "Ch1", 'charge_pC': rep_charge_ch1_pC
                }
                pmt_ch1_fft_data_for_comparison[pmt_id] = {
                    'frequencies': frequencies_ch1[:N_ch1 // 2], 'magnitude': magnitude_spectrum_ch1,
                    'ch_label': "Ch1", 'charge_pC': rep_charge_ch1_pC
                }
            else:
                print(f"    PMT {pmt_id} Ch1: No waveform data for representative event. Skipping comparison data.")
                pmt_ch1_wf_data_for_comparison[pmt_id] = None
                pmt_ch1_fft_data_for_comparison[pmt_id] = None

            if len(wf2_data) > 0:
                N_ch2 = len(wf2_data)
                fft_result_ch2 = scipy.fft.fft(wf2_data)
                frequencies_ch2 = scipy.fft.fftfreq(N_ch2, d=dt_s)
                magnitude_spectrum_ch2 = np.abs(fft_result_ch2[:N_ch2 // 2])
                pmt_ch2_wf_data_for_comparison[pmt_id] = {
                    'x_wf': x_wf, 'wf_data': wf2_data, 'ch_label': "Ch2", 'charge_pC': rep_charge_ch2_pC
                }
                pmt_ch2_fft_data_for_comparison[pmt_id] = {
                    'frequencies': frequencies_ch2[:N_ch2 // 2], 'magnitude': magnitude_spectrum_ch2,
                    'ch_label': "Ch2", 'charge_pC': rep_charge_ch2_pC
                }
            else:
                print(f"    PMT {pmt_id} Ch2: No waveform data for representative event. Skipping comparison data.")
                pmt_ch2_wf_data_for_comparison[pmt_id] = None
                pmt_ch2_fft_data_for_comparison[pmt_id] = None


        print("\n--- Generating Comparison Plots for all PMTs ---")

        plot_multi_pmt_waveforms_comparison(
            pmt_ch1_wf_data_for_comparison,
            f"Time Domain Waveforms: Ch1 (Representative 100 pC Peak Events) - LOM {current_lom_name}",
            f"{lom_output_prefix}all_pmts_time_domain_ch1_comparison", # Base filename
            output_dirs["waveform_analysis_ch1"],
            x_range_mode="peak_focus"
        )

        plot_multi_pmt_waveforms_comparison(
            pmt_ch1_wf_data_for_comparison,
            f"Time Domain Waveforms: Ch1 (Representative 100 pC Peak Events) - LOM {current_lom_name}",
            f"{lom_output_prefix}all_pmts_time_domain_ch1_comparison", # Base filename
            output_dirs["waveform_analysis_ch1"],
            x_range_mode="broad"
        )

        plot_multi_pmt_waveforms_comparison(
            pmt_ch2_wf_data_for_comparison,
            f"Time Domain Waveforms: Ch2 (Representative 100 pC Peak Events) - LOM {current_lom_name}",
            f"{lom_output_prefix}all_pmts_time_domain_ch2_comparison", # Base filename
            output_dirs["waveform_analysis_ch2"],
            x_range_mode="peak_focus"
        )

        plot_multi_pmt_waveforms_comparison(
            pmt_ch2_wf_data_for_comparison,
            f"Time Domain Waveforms: Ch2 (Representative 100 pC Peak Events) - LOM {current_lom_name}",
            f"{lom_output_prefix}all_pmts_time_domain_ch2_comparison", # Base filename
            output_dirs["waveform_analysis_ch2"],
            x_range_mode="broad"
        )

        plot_multi_pmt_ffts_comparison(
            pmt_ch1_fft_data_for_comparison,
            f"Frequency Domain Spectra: Ch1 (Representative 100 pC Peak Events) - LOM {current_lom_name}",
            f"{lom_output_prefix}all_pmts_freq_domain_ch1_comparison.pdf",
            output_dirs["fft_analysis_ch1"]
        )

        plot_multi_pmt_ffts_comparison(
            pmt_ch2_fft_data_for_comparison,
            f"Frequency Domain Spectra: Ch2 (100 pC Peak Events) - LOM {current_lom_name}",
            f"{lom_output_prefix}all_pmts_freq_domain_ch2_comparison.pdf",
            output_dirs["fft_analysis_ch2"]
        )
        print("\n--- FFT Analysis and Comparison Plots Complete ---")

        # --- Muon Charge Distribution Plot for the current LOM ---
        plot_muon_charge_distribution_ch2(pmt_to_infile_map, args.assumed_gain, output_dirs["charge_distributions"], lom_output_prefix)

    # All LOMs processed, optionally show all plots at the end if desired (might be too many windows)
    # plt.show()
    print("\n========================================================")
    print("=== All LOMs Processed. Analysis Complete. ===")
    print("========================================================")