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
# Note: This is an example, actual physical positions might vary.
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
    global_ymax_adc_ch1 = 4200
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
            ax.set_title(f"PMT {pmt_id_to_plot} {data['ch_label']} ({data['charge_pC']:.1f}pC)", fontsize=12)
            ax.set_xlabel("Time (ns)", fontsize=12)
            ax.set_ylabel("ADC Counts", fontsize=12)
            ax.set_xlim([xmin_ns_current, xmax_ns_current])
            ax.set_ylim([ymin_adc, ymax_adc])

            # Add cutting line for Ch1 at ADC 300
            if "Ch1" in plot_title:
                ax.axhline(y=300, color='orange', linestyle='--', linewidth=0.8, alpha=0.7)
                ax.axhline(y=4095, color='purple', linestyle='-', linewidth=0.8, alpha=0.7)
                # No legend label here, as it's typically handled by a general legend or implied
            # Add cutting lines for Ch2 based on new interpretation (raw ADC, high baseline, negative pulse)
            elif "Ch2" in plot_title:
                ax.axhline(y=3800, color='gray', linestyle='-', linewidth=0.8, alpha=0.7, label='Baseline (~3800)')
                ax.axhline(y=3500, color='green', linestyle='--', linewidth=0.8, alpha=0.7, label='ADC Cut (3500)')
                ax.legend(fontsize=12, loc='upper right') # Only legend in Ch2 plots to avoid clutter
        else:
            ax.text(0.5, 0.5, f"PMT {pmt_id_to_plot}\nNo Data in Peak", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_title(f"PMT {pmt_id_to_plot}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

        ax.grid(True, linestyle=':', lw=0.5)
        ax.tick_params(labelsize=15)

    # Hide any unused subplots (if less than 16 PMTs available)
    for j in range(len(custom_subplot_order_for_4x4_grid), 16):
        if j < len(axes): # Ensure axis exists before deleting
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
            ax.set_title(f"PMT {pmt_id_to_plot} {data['ch_label']} ({data['charge_pC']:.1f}pC)", fontsize=12)
            ax.set_xlabel("Frequency (MHz)", fontsize=12)
            ax.set_ylabel("Magnitude", fontsize=12)
            ax.set_xlim([0, global_xmax_freq_mhz])
            ax.set_yscale('log')
        else:
            ax.text(0.5, 0.5, f"PMT {pmt_id_to_plot}\nNo Data in Peak", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_title(f"PMT {pmt_id_to_plot}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

        ax.grid(True, linestyle=':', lw=0.5)
        ax.tick_params(labelsize=15)

    for j in range(len(custom_subplot_order_for_4x4_grid), 16):
        if j < len(axes): # Ensure axis exists before deleting
            fig.delaxes(axes[j])

    fig.suptitle(plot_title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, output_filename))
    plt.close(fig)
    print(f'Saved {os.path.join(output_dir, output_filename)}')


# New function modified to accept vmin/vmax for colorbar and not add colorbar itself
def plot_single_charge_correlation_subplot(ax, df_pmt_events, pmt_full_name, m_slope, c_intercept, r_squared, fit_q1_min_pC, fit_q1_max_pC, plot_type='log', cbar_vmin=None, cbar_vmax=None):
    x_data_all = df_pmt_events['ch1_pC'].values
    y_data_all = df_pmt_events['ch2_pC'].values

    # Define common ranges for these plots as per Chris Wendt's suggestion
    x_plot_min_pC = 0
    x_plot_max_pC = 100 # Keep display Xmax at 100pC for consistency
    y_plot_min_pC = 0
    y_plot_max_pC = 10

    # Determine bins for the 2D histograms (for the new focused ranges)
    x_bins_focused = np.linspace(x_plot_min_pC, x_plot_max_pC, 101) # 1 pC bin size
    y_bins_focused = np.linspace(y_plot_min_pC, y_plot_max_pC, 51)  # 0.2 pC bin size

    if plot_type == 'log':
        h = ax.hist2d(x_data_all, y_data_all, bins=[x_bins_focused, y_bins_focused], cmap='viridis', norm=LogNorm(vmin=cbar_vmin, vmax=cbar_vmax))
        title_suffix = ''
    else: # linear
        h = ax.hist2d(x_data_all, y_data_all, bins=[x_bins_focused, y_bins_focused], cmap='viridis', vmin=cbar_vmin, vmax=cbar_vmax)
        title_suffix = ''
    
    ax.set_title(f'{pmt_full_name} {title_suffix}', fontsize=12) # Smaller title for subplots
    ax.set_xlabel('Charge Ch1 (pC)', fontsize=15) # Smaller labels
    ax.set_ylabel('Charge Ch2 (pC)', fontsize=15)
    ax.set_xlim(x_plot_min_pC, x_plot_max_pC)
    ax.set_ylim(y_plot_min_pC, y_plot_max_pC)
    ax.tick_params(labelsize=15) # Smaller tick labels
    ax.grid(True, linestyle=':', alpha=0.6)

    # Plot the regression line
    if not np.isnan(m_slope):
        # Plot the fit line across the full display range for context
        x_fit_plot = np.linspace(x_plot_min_pC, x_plot_max_pC, 100)
        y_fit_plot = linear_function(x_fit_plot, m_slope, c_intercept)
        ax.plot(x_fit_plot, y_fit_plot, color='red', linestyle='--', linewidth=1.5,
                                 label=f'Fit: Q2 = {m_slope:.2f}*Q1 + {c_intercept:.2f} pC\n($R^2={r_squared:.2f}$)\nFit Range: [{fit_q1_min_pC:.1f}-{fit_q1_max_pC:.1f}] pC')
    ax.legend(fontsize=15, loc='upper left') # Smaller legend

    return h[3], m_slope, c_intercept, r_squared # Return mappable and fit parameters

def plot_charge_correlation_per_pmt(df_pmt_events, pmt_full_name, output_dir, pmt_output_prefix, fit_q1_min_pC, fit_q1_max_pC_for_fit):
    """
    Plots Ch1 vs Ch2 charge correlation for a single PMT, with linear and log scales side-by-side,
    and fixed colorbar ranges. Saves to a single PDF file per PMT.
    Returns the fit parameters (m, c, R^2).
    """
    # Define fixed colorbar ranges as per user request
    LINEAR_CBAR_VMIN = 0
    LINEAR_CBAR_VMAX = 600
    LOG_CBAR_VMIN = 1e0 # 10^0
    LOG_CBAR_VMAX = 1e3 # 10^3

    fig, axes = plt.subplots(1, 2, figsize=(18, 8)) # Landscape orientation for 2 plots side-by-side
    
    m_slope, c_intercept, r_squared = np.nan, np.nan, np.nan
    # Use the specified fitting range for calculation
    df_fit = df_pmt_events[(df_pmt_events['ch1_pC'] >= fit_q1_min_pC) & (df_pmt_events['ch1_pC'] <= fit_q1_max_pC_for_fit)]

    if len(df_fit) >= 2:
        try:
            # Add bounds to force slope > 0 and slope < 1
            # lower bounds for [m, c], upper bounds for [m, c]
            params, cov = curve_fit(linear_function, df_fit['ch1_pC'].values, df_fit['ch2_pC'].values,
                                     bounds=([0.0, -np.inf], [1.0, np.inf])) # 0 < m < 1
            m_slope, c_intercept = params
            residuals = df_fit['ch2_pC'].values - linear_function(df_fit['ch1_pC'].values, m_slope, c_intercept)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((df_fit['ch2_pC'].values - np.mean(df_fit['ch2_pC'].values))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
            print(f"  {pmt_full_name}: Fit (m, c, R^2) = ({m_slope:.2f}, {c_intercept:.2f}, {r_squared:.2f})")
        except RuntimeError:
            warnings.warn(f"Could not fit linear regression for {pmt_full_name} with 0<m<1 constraint. Check data or fit range.")
        except ValueError as e: # Catch cases where bounds make fitting impossible
            warnings.warn(f"Fitting error for {pmt_full_name} with 0<m<1 constraint: {e}. Data might not support positive slope in range.")
    else:
        warnings.warn(f"Not enough valid charge data points in fit range [{fit_q1_min_pC:.1f}, {fit_q1_max_pC_for_fit:.1f}] pC for {pmt_full_name}. Skipping regression.")

    # Plot Linear Scale
    mappable_linear, _, _, _ = plot_single_charge_correlation_subplot(
        axes[0], df_pmt_events, pmt_full_name,
        m_slope, c_intercept, r_squared, fit_q1_min_pC, fit_q1_max_pC_for_fit, plot_type='linear',
        cbar_vmin=LINEAR_CBAR_VMIN, cbar_vmax=LINEAR_CBAR_VMAX
    )
    fig.colorbar(mappable_linear, ax=axes[0], label='Number of Events (Linear Scale)')


    # Plot Log Scale
    mappable_log, _, _, _ = plot_single_charge_correlation_subplot(
        axes[1], df_pmt_events, pmt_full_name,
        m_slope, c_intercept, r_squared, fit_q1_min_pC, fit_q1_max_pC_for_fit, plot_type='log',
        cbar_vmin=LOG_CBAR_VMIN, cbar_vmax=LOG_CBAR_VMAX
    )
    # Define custom ticks for the log colorbar to ensure readability
    log_ticks = [10**i for i in range(int(np.log10(LOG_CBAR_VMIN)), int(np.log10(LOG_CBAR_VMAX)) + 1)]
    fig.colorbar(mappable_log, ax=axes[1], label='Number of Events (Log Scale)', format='%.0e', ticks=log_ticks)

    fig.suptitle(f"Charge Correlation (Ch1 vs Ch2) for {pmt_full_name}", fontsize=15, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust rect for suptitle
    
    output_filename = f"{pmt_output_prefix}charge_correlation_ch1_ch2.pdf"
    plt.savefig(os.path.join(output_dir, output_filename))
    plt.close(fig)
    print(f"Saved Charge Correlation plot for {pmt_full_name} to: {os.path.join(output_dir, output_filename)}")

    return m_slope, c_intercept, r_squared # Return fit parameters


def plot_muon_charge_distribution_ch2_per_pmt(ch2_charges_pC_series, pmt_full_name, charge_per_pe_pC_ch2, output_dir, pmt_output_prefix):
    """
    Plots Ch2 charge distribution in PE for a single PMT, using the calculated charge_per_pe_pC_ch2.
    X-axis in PE, Y-axis in Counts (Log Scale).

    Args:
        ch2_charges_pC_series (pd.Series): Series of Ch2 charges in pC.
        pmt_full_name (str): Full name of the PMT (e.g., "LOM16-01: PMT 00").
        charge_per_pe_pC_ch2 (float): The calculated charge (in pC) corresponding to one photoelectron for Ch2.
        output_dir (str): Directory to save the output PDF.
        pmt_output_prefix (str): Prefix for the output filename.
    """
    if charge_per_pe_pC_ch2 <= 0 or np.isnan(charge_per_pe_pC_ch2):
        warnings.warn(f"Invalid or non-positive charge_per_pe_pC_ch2 ({charge_per_pe_pC_ch2:.4f} pC/PE) for {pmt_full_name}. Cannot plot charge distribution in PE. Skipping.")
        return

    charges_pe = ch2_charges_pC_series.values / charge_per_pe_pC_ch2
    charges_pe = charges_pe[~np.isnan(charges_pe)]

    if charges_pe.size == 0:
        print(f"  No valid Ch2 charge data for distribution for {pmt_full_name}. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x_dist_min_pe = 0
    x_dist_max_pe = 160 # Keep consistent with previous general distribution plot
    bins = np.linspace(x_dist_min_pe, x_dist_max_pe, 81) # 0.25 PE bin size for 0-20 PE

    ax.hist(charges_pe, bins=bins, color='skyblue', edgecolor='blue', alpha=0.7)

    ax.set_xlabel('Charge (PE)', fontsize=12)
    ax.set_ylabel('Counts', fontsize=12) # User requested Log Scale
    ax.set_title(f'Muon Charge Distribution: Ch2 - {pmt_full_name}', fontsize=14)
    
    ax.set_xlim([x_dist_min_pe, x_dist_max_pe])
    ax.set_yscale('log') # Set Y-axis to Log Scale as requested

    ax.tick_params(labelsize=15)
    ax.grid(True, linestyle=':', alpha=0.6)

    output_filename = f"{pmt_output_prefix}muon_charge_distribution_ch2.pdf"
    plt.savefig(os.path.join(output_dir, output_filename))
    plt.close(fig)
    print(f"Saved Muon Charge Distribution (Ch2) plot for {pmt_full_name} to: {os.path.join(output_dir, output_filename)}")


def plot_event_counts_and_export_excel(pmt_to_infile_map, lom_name, output_dir, lom_output_prefix, threshold_pC):
    """
    Plots Number of Events vs PMT No. for a given LOM at a specific charge threshold
    and exports the data to an XLSX file. This is a combined bar chart for the LOM.
    """
    event_counts_data = []
    pmt_ids_sorted = sorted(pmt_to_infile_map.keys())

    print(f"\n--- Calculating Event Counts for {lom_name} (Threshold: {threshold_pC:.1f} pC) ---")
    for pmt_id in pmt_ids_sorted:
        infile_pmt = pmt_to_infile_map[pmt_id]
        timestamps, _ = get_timestamps(infile_pmt, min_charge_pC=threshold_pC, max_charge_pC=float('inf'))
        event_counts_data.append({'LOM ID': lom_name, 'PMT ID': pmt_id, 'Number of Events': len(timestamps)})
        print(f"  {lom_name} PMT {pmt_id:02d}: {len(timestamps)} events")

    df_event_counts = pd.DataFrame(event_counts_data)

    # Export to XLSX
    xlsx_filename = f"{lom_output_prefix}event_counts_Th_{threshold_pC:.1f}pC.xlsx"
    xlsx_path = os.path.join(output_dir, xlsx_filename)
    df_event_counts.to_excel(xlsx_path, index=False)
    print(f"Exported event counts for {lom_name} to: {xlsx_path}")

    # Plotting Number of Events vs PMT No. (as a single combined bar chart for the LOM)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(df_event_counts['PMT ID'], df_event_counts['Number of Events'], color='skyblue', edgecolor='blue')
    ax.set_xlabel('PMT No.', fontsize=12)
    ax.set_ylabel('Number of Events', fontsize=15)
    ax.set_title(f'Number of Events per PMT - {lom_name} (Threshold: {threshold_pC:.1f} pC)', fontsize=15)
    ax.set_xticks(df_event_counts['PMT ID'])
    ax.tick_params(labelsize=15)
    ax.grid(axis='y', linestyle=':', alpha=0.6)

    plot_filename = f"{lom_output_prefix}event_counts_Th_{threshold_pC:.1f}pC.pdf"
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.close(fig)
    print(f"Saved event counts plot for {lom_name} to: {os.path.join(output_dir, plot_filename)}")


# --- NEW: Functions for multi-subplot (16 PMTs) plots ---

def plot_multi_pmt_charge_correlations_overview(pmt_dfs_for_correlation, lom_name, output_dir, lom_output_prefix, fit_q1_min_pC, fit_q1_max_pC_for_fit):
    """
    Plots Charge Correlation (Ch1 vs Ch2) for all 16 PMTs of a LOM in a single 4x4 figure.
    Also collects and exports fitting parameters to XLSX and a text summary.
    """
    fig, axes = plt.subplots(4, 4, figsize=(20, 16)) # 4x4 grid for 16 subplots
    axes = axes.flatten()

    LINEAR_CBAR_VMIN = 0
    LINEAR_CBAR_VMAX = 600
    LOG_CBAR_VMIN = 1e0
    LOG_CBAR_VMAX = 1e3

    # Data to store for XLSX export
    fitting_params_data = []
    fitting_equations_summary = [] # For text file output

    # Custom PMT order for plotting (matching physical layout if available)
    # This order is based on example given in previous turns, adjust if actual physical map is different
    custom_subplot_order = np.array([8, 10, 12, 14, 0, 2, 4, 6, 1, 3, 5, 7, 9, 11, 13, 15])  
    
    # Ensure all PMTs are included in a consistent order, using the actual data's PMT IDs
    pmt_ids_present = sorted(pmt_dfs_for_correlation.keys())
    pmt_ids_to_plot_ordered = [pid for pid in custom_subplot_order if pid in pmt_ids_present]
    if len(pmt_ids_to_plot_ordered) < 16: # Fallback if map doesn't contain all 16 or custom order is problematic
        pmt_ids_to_plot_ordered = sorted(pmt_dfs_for_correlation.keys()) # Use all available and sorted

    main_mappable = None # To store a mappable for the common colorbar

    fitting_equations_summary.append(f"--- Fitting Equations Summary for {lom_name} ---")

    for i, pmt_id in enumerate(pmt_ids_to_plot_ordered):
        ax = axes[i]
        df_pmt_events = pmt_dfs_for_correlation.get(pmt_id)
        pmt_full_name = f"PMT {pmt_id:02d}" # For subplot title

        m_slope, c_intercept, r_squared = np.nan, np.nan, np.nan

        if df_pmt_events is not None and not df_pmt_events.empty:
            df_fit = df_pmt_events[(df_pmt_events['ch1_pC'] >= fit_q1_min_pC) & (df_pmt_events['ch1_pC'] <= fit_q1_max_pC_for_fit)]

            if len(df_fit) >= 2:
                try:
                    # Add bounds to force slope > 0 and slope < 1
                    # lower bounds for [m, c], upper bounds for [m, c]
                    params, cov = curve_fit(linear_function, df_fit['ch1_pC'].values, df_fit['ch2_pC'].values,
                                             bounds=([0.0, -np.inf], [1.0, np.inf])) # 0 < m < 1
                    m_slope, c_intercept = params
                    residuals = df_fit['ch2_pC'].values - linear_function(df_fit['ch1_pC'].values, m_slope, c_intercept)
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((df_fit['ch2_pC'].values - np.mean(df_fit['ch2_pC'].values))**2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
                except RuntimeError:
                    warnings.warn(f"Could not fit linear regression for {pmt_full_name} with 0<m<1 constraint. Check data or fit range.")
                    pass # Keep nan values if fit fails
                except ValueError as e: # Catch cases where bounds make fitting impossible
                    warnings.warn(f"Fitting error for {pmt_full_name} with 0<m<1 constraint: {e}. Data might not support positive slope in range.")
                    pass # Keep nan values
            
            # Plot as Log Scale for overview (as requested for combined view)
            mappable, _, _, _ = plot_single_charge_correlation_subplot(
                ax, df_pmt_events, pmt_full_name,
                m_slope, c_intercept, r_squared, fit_q1_min_pC, fit_q1_max_pC_for_fit, plot_type='log',
                cbar_vmin=LOG_CBAR_VMIN, cbar_vmax=LOG_CBAR_VMAX
            )
            if main_mappable is None: # Store only one mappable to pass to the common colorbar
                 main_mappable = mappable
        else:
            ax.text(0.5, 0.5, f"{pmt_full_name}\nNo Data", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_title(pmt_full_name, fontsize=15)
            ax.set_xticks([])
            ax.set_yticks([])

        fitting_params_data.append({
            'LOM ID': lom_name,
            'PMT ID': pmt_id,
            'Slope (m)': m_slope,
            'Intercept (c)': c_intercept,
            'R-squared': r_squared,
            'Fit Range Min (pC)': fit_q1_min_pC,
            'Fit Range Max (pC)': fit_q1_max_pC_for_fit
        })
        
        # Format and add to summary text
        if not np.isnan(m_slope):
            eq_str = f"Q2 = {m_slope:.4f} * Q1 + {c_intercept:.4f} pC ($R^2={r_squared:.4f}$)"
            fitting_equations_summary.append(f"{lom_name} PMT {pmt_id:02d}: {eq_str}")
        else:
            fitting_equations_summary.append(f"{lom_name} PMT {pmt_id:02d}: No valid fit.")

        ax.grid(True, linestyle=':', lw=0.5)
        ax.tick_params(labelsize=15)

    # Add a single colorbar for the whole figure
    if main_mappable is not None:
        # Create an axis explicitly to the right of the whole grid
        cbar_ax_pos = [0.92, 0.1, 0.02, 0.8] # [left, bottom, width, height] relative to figure
        cbar_ax = fig.add_axes(cbar_ax_pos)
        fig.colorbar(main_mappable, cax=cbar_ax, label='Number of Events', format='%.0e', ticks=[10**i for i in range(int(np.log10(LOG_CBAR_VMIN)), int(np.log10(LOG_CBAR_VMAX)) + 1)])


    fig.suptitle(f"Charge Correlation Overview (Ch1 vs Ch2) - {lom_name}", fontsize=18, y=1.0)
    plt.tight_layout(rect=[0, 0.03, 0.9, 0.96]) # Adjust rect to make space for custom colorbar
    output_filename = f"{lom_output_prefix}all_pmts_charge_correlation_overview.pdf"
    plt.savefig(os.path.join(output_dir, output_filename))
    plt.close(fig)
    print(f'Saved all PMTs Charge Correlation Overview for {lom_name} to: {os.path.join(output_dir, output_filename)}')

    # Export fitting parameters to XLSX
    df_fitting_params = pd.DataFrame(fitting_params_data)
    xlsx_filename = f"{lom_output_prefix}charge_correlation_fitting_params.xlsx"
    xlsx_path = os.path.join(output_dir, xlsx_filename)
    df_fitting_params.to_excel(xlsx_path, index=False)
    print(f"Exported Charge Correlation Fitting Parameters for {lom_name} to: {xlsx_path}")

    # Export fitting equations summary to a text file
    txt_filename = f"{lom_output_prefix}charge_correlation_equations_summary.txt"
    txt_path = os.path.join(output_dir, txt_filename)
    with open(txt_path, 'w') as f:
        for line in fitting_equations_summary:
            f.write(line + '\n')
    print(f"Exported Charge Correlation Equations Summary for {lom_name} to: {txt_path}")

    # Return the dataframe of fitting parameters for potential use in subsequent functions (e.g., for PE conversion)
    return df_fitting_params


def plot_multi_pmt_ch2_distributions_overview(pmt_ch2_data_per_lom, pmt_fitting_slopes, lom_name, assumed_ch1_pc_per_pe, output_dir, lom_output_prefix):
    """
    Plots Ch2 Charge Distribution in PE for all 16 PMTs of a LOM in a single 4x4 figure.
    Uses fitting slopes from charge correlation to convert pC to PE for each PMT.
    Y-axis in Counts (Log Scale).

    Args:
        pmt_ch2_data_per_lom (dict): Dict {pmt_id: pd.Series of Ch2 charges in pC}.
        pmt_fitting_slopes (pd.DataFrame): DataFrame containing 'PMT ID' and 'Slope (m)' from charge correlation.
        lom_name (str): Name of the LOM.
        assumed_ch1_pc_per_pe (float): Assumed pC per PE for Channel 1.
        output_dir (str): Directory to save the output PDF.
        lom_output_prefix (str): Prefix for the output filename.
    """
    fig, axes = plt.subplots(4, 4, figsize=(20, 16)) # 4x4 grid for 16 subplots
    axes = axes.flatten()

    electron_charge_C = 1.602e-19 # This constant is not used if we convert via slope and assumed_ch1_pc_per_pe
    
    x_dist_min_pe = 0
    x_dist_max_pe = 160 # Keep consistent with previous general distribution plot
    bins = np.linspace(x_dist_min_pe, x_dist_max_pe, 81) # 0.25 PE bin size for 0-20 PE

    # Create a lookup for slopes
    slope_lookup = pmt_fitting_slopes.set_index('PMT ID')['Slope (m)'].to_dict()

    # Custom PMT order for plotting (matching physical layout if available)
    custom_subplot_order = np.array([8, 10, 12, 14, 0, 2, 4, 6, 1, 3, 5, 7, 9, 11, 13, 15])  

    # Ensure all PMTs are included in a consistent order, using the actual data's PMT IDs
    pmt_ids_present = sorted(pmt_ch2_data_per_lom.keys())
    pmt_ids_to_plot_ordered = [pid for pid in custom_subplot_order if pid in pmt_ids_present]
    if len(pmt_ids_to_plot_ordered) < 16: # Fallback if map doesn't contain all 16 or custom order is problematic
        pmt_ids_to_plot_ordered = sorted(pmt_ch2_data_per_lom.keys()) # Use all available and sorted

    for i, pmt_id in enumerate(pmt_ids_to_plot_ordered):
        ax = axes[i]
        ch2_charges_pC_series = pmt_ch2_data_per_lom.get(pmt_id)
        pmt_full_name = f"PMT {pmt_id:02d}" # For subplot title

        slope_m = slope_lookup.get(pmt_id, np.nan) # Get the slope for this specific PMT
        
        # Calculate Ch2 charge per PE using the slope and assumed Ch1 pC/PE
        # Charge_per_PE_pC_Ch2 = Slope (pC_Ch2/pC_Ch1) * Assumed_Ch1_pC_per_PE
        charge_per_pe_pC_ch2 = slope_m * assumed_ch1_pc_per_pe if not np.isnan(slope_m) else np.nan

        if ch2_charges_pC_series is not None and not ch2_charges_pC_series.empty and \
           not np.isnan(charge_per_pe_pC_ch2) and charge_per_pe_pC_ch2 > 0:
            
            charges_pe = ch2_charges_pC_series.values / charge_per_pe_pC_ch2
            charges_pe = charges_pe[~np.isnan(charges_pe)]

            if charges_pe.size > 0:
                ax.hist(charges_pe, bins=bins, color='skyblue', edgecolor='blue', alpha=0.7)
                ax.set_title(pmt_full_name, fontsize=12)
                ax.set_xlabel('Charge (PE)', fontsize=12)
                ax.set_ylabel('Counts', fontsize=12)
                ax.set_xlim([x_dist_min_pe, x_dist_max_pe])
                ax.set_yscale('log')
                ax.text(0.95, 0.95, f'1PE = {charge_per_pe_pC_ch2:.3f} pC',
                        transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.5))
            else:
                ax.text(0.5, 0.5, f"{pmt_full_name}\nNo Valid PE Data", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10, color='gray')
                ax.set_title(pmt_full_name, fontsize=15)
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            ax.text(0.5, 0.5, f"{pmt_full_name}\nNo Data or Invalid Gain", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_title(pmt_full_name, fontsize=15)
            ax.set_xticks([])
            ax.set_yticks([])

        ax.tick_params(labelsize=15)
        ax.grid(True, linestyle=':', lw=0.5)

    fig.suptitle(f"Ch2 Charge Distribution Overview - {lom_name} (Charge in PE, calculated via Correlation Slope)", fontsize=18, y=1.0)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust rect for suptitle
    output_filename = f"{lom_output_prefix}all_pmts_ch2_distribution_overview_from_correlation.pdf"
    plt.savefig(os.path.join(output_dir, output_filename))
    plt.close(fig)
    print(f'Saved all PMTs Ch2 Distribution Overview (in PE, from correlation) for {lom_name} to: {os.path.join(output_dir, output_filename)}')


def plot_multi_pmt_event_counts_overview(pmt_event_counts_data, lom_name, output_dir, lom_output_prefix, threshold_pC):
    """
    Plots Event Counts for all 16 PMTs of a LOM in a single 4x4 figure as bar charts.
    Each subplot shows a single bar for the respective PMT's event count.
    """
    fig, axes = plt.subplots(4, 4, figsize=(18, 14)) # 4x4 grid for 16 subplots
    axes = axes.flatten()

    # Custom PMT order for plotting
    custom_subplot_order = np.array([8, 10, 12, 14, 0, 2, 4, 6, 1, 3, 5, 7, 9, 11, 13, 15])  
    
    # Ensure all PMTs are included in a consistent order
    pmt_ids_present = sorted([d['PMT ID'] for d in pmt_event_counts_data])
    pmt_ids_to_plot_ordered = [pid for pid in custom_subplot_order if pid in pmt_ids_present]
    if len(pmt_ids_to_plot_ordered) < 16: # Fallback if map doesn't contain all 16 or custom order is problematic
        pmt_ids_to_plot_ordered = pmt_ids_present # Use all available and sorted

    # Convert list of dicts to dict for easy lookup
    pmt_counts_dict = {d['PMT ID']: d['Number of Events'] for d in pmt_event_counts_data}
    
    global_max_counts = max(pmt_counts_dict.values()) if pmt_counts_dict else 100 # Adjust Ylim consistently

    for i, pmt_id in enumerate(pmt_ids_to_plot_ordered):
        ax = axes[i]
        counts = pmt_counts_dict.get(pmt_id, 0) # Get count for this PMT, default to 0 if not found
        pmt_full_name = f"PMT {pmt_id:02d}"

        if counts > 0:
            # Plot as a single bar in its subplot
            ax.bar([''], [counts], color='skyblue', edgecolor='blue') # Use [''] for single bar on x-axis
            ax.set_title(f'{pmt_full_name} ({counts} events)', fontsize=10)
            ax.set_ylabel('Events', fontsize=12)
            ax.set_ylim(0, global_max_counts * 1.1) # Set consistent Y-limit
            ax.set_xticks([]) # Remove x-ticks as there's only one bar
        else:
            ax.text(0.5, 0.5, f"{pmt_full_name}\nNo Events > {threshold_pC:.1f}pC", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_title(pmt_full_name, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            
        ax.tick_params(labelsize=15)
        ax.grid(axis='y', linestyle=':', lw=0.5)

    fig.suptitle(f"Event Counts per PMT Overview - {lom_name} (Threshold: {threshold_pC:.1f} pC)", fontsize=18, y=1.0)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    output_filename = f"{lom_output_prefix}all_pmts_event_counts_overview.pdf"
    plt.savefig(os.path.join(output_dir, output_filename))
    plt.close(fig)
    print(f'Saved all PMTs Event Counts Overview for {lom_name} to: {os.path.join(output_dir, output_filename)}')


# --- End: Plotting Functions ---


if __name__ == "__main__":
    # Configure command-line argument parser
    parser = argparse.ArgumentParser(description="Streamlined Muon Analysis Script: Focuses on coincidence and FFT analysis.")
    parser.add_argument("lom_data_dirs", nargs='+', help="Input LOM data directories (e.g., 'path/to/LOM16-01/'). Each directory should contain 16 PMT HDF files.", type=str)
    parser.add_argument("--output_dir", "-o", help="Output directory for plots (default: ./plots_output)", type=str, default='./plots_output')
    parser.add_argument("--assumed_gain", type=float, default=5e6, help="Assumed PMT gain for PE conversion (electrons/PE). This is a general gain value, the Ch2 PE conversion for distribution plot will use the slope from correlation.")
    parser.add_argument("--fit_q1_min", type=float, default=5.0, help="Minimum Ch1 charge (pC) for linear regression in correlation plot.")
    # The fit_q1_max will be explicitly set to 67.2 pC for the 96 plots
    parser.add_argument("--fit_q1_max", type=float, default=100.0, help="Maximum Ch1 charge (pC) for linear regression in correlation plot (general, will be overridden for 96 plots).")
    parser.add_argument("--waveform_sample_size", type=int, default=5, help="Number of well-correlated and outlier waveforms to sample and plot.")
    args = parser.parse_args()

    # Define the fixed time window as requested by Yuya
    # This will be used for all coincidence calculations
    FIXED_COINCIDENCE_WINDOW_NS = 100.0

    # Define the coincidence thresholds as before
    COINCIDENCE_THRESHOLDS_PC = [10.0, 20.0, 50.0, 100.0]

    # Specific threshold for Event Count Plot
    EVENT_COUNT_THRESHOLD_PC = 10.0

    # Fixed fitting range for 96 Charge Correlation plots
    FIXED_CORRELATION_FIT_Q1_MAX_PC = 67.2

    # Assumed pC/PE for Channel 1 (given by Yuya) - This is critical for Ch2 PE conversion
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
        "event_counts": os.path.join(output_base_dir, "event_counts"), # NEW: for event count plots and excel
        "waveform_samples": os.path.join(output_base_dir, "waveform_samples") # For sampled waveforms
    }
    for d in output_dirs.values():
        os.makedirs(d, exist_ok=True)
    print(f"Output plots will be saved to: {output_base_dir} and its subfolders.")

    # --- Prepare combined Coincidence Matrices figures (These remain as before, overall view) ---
    num_loms_to_process = len(sorted(args.lom_data_dirs)) # Use actual number of LOMs from input

    # Linear Scale Coincidence Matrices for all LOMs combined figure
    gs_matrices_linear = GridSpec(num_loms_to_process, len(COINCIDENCE_THRESHOLDS_PC) + 1, width_ratios=[1]*len(COINCIDENCE_THRESHOLDS_PC) + [0.05], wspace=0.1, hspace=0.3)
    fig_matrices_all_loms_linear = plt.figure(figsize=(6 * len(COINCIDENCE_THRESHOLDS_PC) + 3, 5.5 * num_loms_to_process))
    fig_matrices_all_loms_linear.suptitle(f"Coincidence Rate Matrices (Linear Scale) for All LOMs\nFixed ΔT: {FIXED_COINCIDENCE_WINDOW_NS} ns", fontsize=16, y=0.99)
    matrices_linear_mappables = []

    # Log Scale Coincidence Matrices for all LOMs combined figure
    gs_matrices_log = GridSpec(num_loms_to_process, len(COINCIDENCE_THRESHOLDS_PC) + 1, width_ratios=[1]*len(COINCIDENCE_THRESHOLDS_PC) + [0.05], wspace=0.1, hspace=0.3)
    fig_matrices_all_loms_log = plt.figure(figsize=(6 * len(COINCIDENCE_THRESHOLDS_PC) + 3, 5.5 * num_loms_to_process))
    fig_matrices_all_loms_log.suptitle(f"Coincidence Rate Matrices (Log Scale) for All LOMs\nFixed ΔT: {FIXED_COINCIDENCE_WINDOW_NS} ns", fontsize=16, y=0.99)
    matrices_log_mappables = []

    # Define fixed colorbar ticks for linear scale (for coincidence matrices)
    fixed_cbar_ticks = np.array([0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00])
    fixed_cbar_vmax = fixed_cbar_ticks[-1] # Max value for colorbar


    # --- Start: Main loop for LOM-by-LOM processing ---
    sorted_lom_data_dirs = sorted(args.lom_data_dirs)

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

        # --- NEW: Plot Number of Events vs PMT No. and Export to XLSX (combined bar chart per LOM) ---
        # This generates one PDF and one XLSX per LOM with a combined bar chart.
        plot_event_counts_and_export_excel(
            pmt_to_infile_map,
            current_lom_name,
            output_dirs["event_counts"],
            lom_output_prefix,
            EVENT_COUNT_THRESHOLD_PC # 10 pC threshold for event counts
        )

        # --- Data collection for all PMT plots for current LOM (both individual and overview) ---
        pmt_dfs_for_correlation_overview = {} # Store DFs for each PMT for correlation overview plot
        pmt_ch2_data_for_distribution_overview = {} # Store Ch2 series for each PMT for distribution overview plot
        all_pmt_event_counts_for_overview = [] # Store event counts for overview plot (needed for its dedicated overview plot)

        print(f"\n--- Collecting data for PMT-specific plots in {current_lom_name} ---")
        pmt_ids_with_data_sorted = sorted(pmt_to_infile_map.keys())

        # Dictionary to store the fitted slopes for each PMT
        pmt_slopes_for_pe_conversion = {}

        for pmt_id in pmt_ids_with_data_sorted:
            infile_pmt = pmt_to_infile_map[pmt_id]
            pmt_full_name = f"{current_lom_name}: PMT {pmt_id:02d}"
            pmt_output_prefix = f"{lom_output_prefix}PMT{pmt_id:02d}_"

            # Prepare data for this specific PMT (Charge Correlation & Distribution)
            pmt_events_data = []
            data_pmt = read_hdffile(infile_pmt)
            if data_pmt and "data" in data_pmt and "metadata" in data_pmt:
                try:
                    raw_charges_ch1_adc = data_pmt["data"]["charge_ch1"][()]
                    conversion_ch1 = data_pmt["metadata"]["conversion_ch1"][()]
                    raw_charges_ch2_adc = data_pmt["data"]["charge_ch2"][()]
                    conversion_ch2 = data_pmt["metadata"]["conversion_ch2"][()]
                    fpga_time = data_pmt["data"]["FPGAtime"][()]

                    conversion_factor_pC = (1e-6 * (1/60e6) * 1e12)
                    charges_ch1_pC = raw_charges_ch1_adc * (conversion_ch1 * conversion_factor_pC)
                    charges_ch2_pC = raw_charges_ch2_adc * (conversion_ch2 * conversion_factor_pC)

                    min_len_pmt = min(len(charges_ch1_pC), len(charges_ch2_pC), len(fpga_time))
                    for i in range(min_len_pmt):
                        pmt_events_data.append({
                            'ch1_pC': charges_ch1_pC[i],
                            'ch2_pC': charges_ch2_pC[i],
                            'filename': infile_pmt,
                            'timestamp': fpga_time[i],
                            'event_idx': i,
                            'pmt_id': pmt_id
                        })
                except KeyError as e:
                    warnings.warn(f"Missing data for {pmt_full_name}: {e}. Skipping for correlation/distribution.")
            df_pmt_events = pd.DataFrame(pmt_events_data)
            df_pmt_events = df_pmt_events.dropna(subset=['ch1_pC', 'ch2_pC'])
            df_pmt_events = df_pmt_events[(df_pmt_events['ch1_pC'] > 1) & (df_pmt_events['ch2_pC'] > 1)] # Filter out very low charge

            # Store data for multi-PMT overview plots
            pmt_dfs_for_correlation_overview[pmt_id] = df_pmt_events.copy()
            if 'ch2_pC' in df_pmt_events.columns:
                pmt_ch2_data_for_distribution_overview[pmt_id] = df_pmt_events['ch2_pC'].copy()
            
            # Count events for overview plot (this data is also used by plot_event_counts_and_export_excel)
            timestamps_for_count, _ = get_timestamps(infile_pmt, min_charge_pC=EVENT_COUNT_THRESHOLD_PC, max_charge_pC=float('inf'))
            all_pmt_event_counts_for_overview.append({'PMT ID': pmt_id, 'Number of Events': len(timestamps_for_count)})


            # --- Plot Charge Correlation for this PMT (individual file for each of 96 plots) ---
            if not df_pmt_events.empty:
                m_slope_pmt, c_intercept_pmt, r_squared_pmt = plot_charge_correlation_per_pmt(
                    df_pmt_events,
                    pmt_full_name,
                    output_dirs["charge_correlations"],
                    pmt_output_prefix,
                    args.fit_q1_min, # Use the user-defined min for fitting
                    FIXED_CORRELATION_FIT_Q1_MAX_PC # Use the fixed max for fitting (67.2 pC)
                )
                pmt_slopes_for_pe_conversion[pmt_id] = m_slope_pmt # Store the slope
            else:
                print(f"  No valid events for Charge Correlation plot for {pmt_full_name}. Skipping.")
                pmt_slopes_for_pe_conversion[pmt_id] = np.nan # Store NaN if no fit

            # --- Plot Charge Distribution for this PMT (Ch2, Log Scale Y-axis, individual file for each of 96 plots) ---
            if 'ch2_pC' in df_pmt_events.columns and not df_pmt_events['ch2_pC'].empty:
                # Calculate charge_per_pe_pC_ch2 using the fitted slope
                calculated_charge_per_pe_pC_ch2 = pmt_slopes_for_pe_conversion.get(pmt_id, np.nan) * ASSUMED_CH1_PC_PER_PE
                
                plot_muon_charge_distribution_ch2_per_pmt(
                    df_pmt_events['ch2_pC'],
                    pmt_full_name,
                    calculated_charge_per_pe_pC_ch2, # Pass the calculated pC/PE for Ch2
                    output_dirs["charge_distributions"],
                    pmt_output_prefix
                )
            else:
                print(f"  No valid Ch2 charge data for distribution plot for {pmt_full_name}. Skipping.")

        # --- End of per-PMT loop for individual plots ---

        # Convert pmt_slopes_for_pe_conversion to a DataFrame for plot_multi_pmt_ch2_distributions_overview
        df_pmt_slopes_for_overview = pd.DataFrame([{'PMT ID': p_id, 'Slope (m)': slope} 
                                                   for p_id, slope in pmt_slopes_for_pe_conversion.items()])


        # --- NEW: Plot Multi-PMT Overviews for current LOM (16 subplots in one PDF) ---
        print(f"\n--- Generating Multi-PMT Overview Plots for {current_lom_name} (16 Subplots) ---")

        # Plot all PMTs Charge Correlation on one page
        # This function already exports fitting parameters to XLSX and text summary
        df_fitting_params_current_lom = plot_multi_pmt_charge_correlations_overview(
            pmt_dfs_for_correlation_overview,
            current_lom_name,
            output_dirs["charge_correlations"],
            lom_output_prefix,
            args.fit_q1_min,
            FIXED_CORRELATION_FIT_Q1_MAX_PC
        )

        # Plot all PMTs Ch2 Distribution on one page, using the slopes from the correlation fit
        plot_multi_pmt_ch2_distributions_overview(
            pmt_ch2_data_for_distribution_overview,
            df_fitting_params_current_lom, # Pass the DataFrame of fitted slopes
            current_lom_name,
            ASSUMED_CH1_PC_PER_PE, # Pass the assumed Ch1 pC/PE
            output_dirs["charge_distributions"],
            lom_output_prefix
        )

        # Plot all PMTs Event Counts on one page
        plot_multi_pmt_event_counts_overview(
            all_pmt_event_counts_for_overview, # Pass the list of dictionaries
            current_lom_name,
            output_dirs["event_counts"],
            lom_output_prefix,
            EVENT_COUNT_THRESHOLD_PC
        )


        # --- Coincidence Analysis block for current LOM (remains as is) ---
        print("\n--- Starting Coincidence Analysis ---")

        all_pmt_event_times = {}
        all_pmt_livetimes = {}

        print("Calculating Livetimes and Filtering Events for Coincidence (for all PMTs)...")
        pmt_ids_from_map_sorted = sorted(pmt_to_infile_map.keys())

        if not pmt_ids_from_map_sorted:
            print(f"No valid PMT data for LOM {current_lom_name}. Skipping coincidence analysis.")
            continue

        for pmt_id in pmt_ids_from_map_sorted:
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

        lom18_x_indices_filtered = [p for p in lom18_x_labels_order if p in pmt_ids_from_map_sorted]
        lom18_y_indices_filtered = [p for p in lom18_y_labels_order if p in pmt_ids_from_map_sorted]

        if not lom18_x_indices_filtered or not lom18_y_indices_filtered:
            print(f"Not enough PMT data for LOM {current_lom_name} to create LOM18 style coincidence matrix plots. Skipping.")
        else:
            # Loop for Coincidence Matrices (for plotting into the combined figures)
            for c_idx, current_threshold_pC in enumerate(COINCIDENCE_THRESHOLDS_PC):
                print(f"\n--- Building Coincidence Matrix (Threshold: {current_threshold_pC:.1f} pC, Window: {FIXED_COINCIDENCE_WINDOW_NS} ns) ---")

                current_pmt_event_times_filtered = {}
                for pmt_id in pmt_ids_from_map_sorted:
                    infile_pmt = pmt_to_infile_map[pmt_id]
                    filtered_ts_current_th, _ = get_timestamps(infile_pmt, min_charge_pC=current_threshold_pC, max_charge_pC=float('inf'))
                    current_pmt_event_times_filtered[pmt_id] = np.sort(np.array(filtered_ts_current_th))

                max_pmt_id_in_data = max(pmt_ids_from_map_sorted) if pmt_ids_from_map_sorted else -1

                if max_pmt_id_in_data == -1:
                    print("No valid PMT data found to build coincidence matrix. Skipping.")
                    continue

                coincidence_matrix_counts = np.zeros((max_pmt_id_in_data + 1, max_pmt_id_in_data + 1), dtype=int)
                coincidence_matrix_rates = np.zeros((max_pmt_id_in_data + 1, max_pmt_id_in_data + 1), dtype=float)

                for i_pmt in pmt_ids_from_map_sorted:
                    for j_pmt in pmt_ids_from_map_sorted:
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

                # --- Plotting to the COMBINED Coincidence Matrix figures (Linear Scale) ---
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
                ax_matrix_sub_linear_combined.set_xticklabels(lom18_x_indices_filtered, fontsize=12)
                ax_matrix_sub_linear_combined.set_yticklabels(lom18_y_indices_filtered, fontsize=12)

                ax_matrix_sub_linear_combined.set_xlabel('Ch ID', fontsize=12) # Shorter label
                ax_matrix_sub_linear_combined.set_ylabel('Ch ID', fontsize=12) # Shorter label
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

                # --- Plotting to the COMBINED Coincidence Matrix figures (Log Scale) ---
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
                ax_matrix_sub_log_combined.set_title(f"LOM {current_lom_name}\nTh: {current_threshold_pC:.1f} pC", fontsize=12)

                ax_matrix_sub_log_combined.invert_yaxis()
                ax_matrix_sub_log_combined.set_aspect('equal', adjustable='box')

                try: # Add Hemisphere Dividing Lines and Labels for each subplot
                    split_idx_x_for_0 = lom18_x_indices_filtered.index(0)
                    split_idx_y_for_0 = lom18_y_indices_filtered.index(0)

                    ax_matrix_sub_log_combined.axvline(split_idx_x_for_0 + 0.5, color='black', linestyle='-', linewidth=1.5)
                    ax_matrix_sub_log_combined.axhline(split_idx_y_for_0 + 0.5, color='black', linestyle='-', linewidth=1.5)
                except ValueError:
                    warnings.warn("PMT 0 not found in filtered axis order. Cannot draw hemisphere lines/labels precisely for current plot.")
                
                # --- Plotting Total Coincidence Rate (Linear Scale) for individual LOMs ---
                fig_rates_total_per_lom, ax_rates_total_per_lom = plt.subplots(figsize=(6, 4))
                total_coincidence_rates_per_pmt = np.sum(coincidence_matrix_rates, axis=1)
                pmts_to_plot_rate_sorted = sorted([p for p in pmt_ids_from_map_sorted if p <= max_pmt_id_in_data])
                rates_to_plot_sorted = [total_coincidence_rates_per_pmt[p] for p in pmts_to_plot_rate_sorted]
                ax_rates_total_per_lom.plot(pmts_to_plot_rate_sorted, rates_to_plot_sorted, color='blue', marker='o', linestyle='-', markersize=4, label='Data')
                if len(pmts_to_plot_rate_sorted) > 2:
                    try:
                        popt, pcov = curve_fit(polynomial_function, np.array(rates_to_plot_sorted), np.array(rates_to_plot_sorted))
                        x_fit_poly = np.linspace(min(pmts_to_plot_rate_sorted), max(pmts_to_plot_rate_sorted), 100)
                        y_fit_poly = polynomial_function(x_fit_poly, *popt)
                        ax_rates_total_per_lom.plot(x_fit_poly, y_fit_poly, color='red', linestyle='--', label='Polynomial Fit')
                    except RuntimeError:
                        warnings.warn(f"Could not fit polynomial curve for LOM {current_lom_name}, Th {current_threshold_pC:.1f} pC.")
                ax_rates_total_per_lom.legend(fontsize=12)
                ax_rates_total_per_lom.set_xlabel('PMT Channel ID',fontsize=12)
                ax_rates_total_per_lom.set_ylabel('Total Rate [1/s]',fontsize=12)
                ax_rates_total_per_lom.set_title(f"Total Coincidence Rate - LOM {current_lom_name}\nTh: {current_threshold_pC:.1f} pC, dT: {FIXED_COINCIDENCE_WINDOW_NS}ns", fontsize=9)
                ax_rates_total_per_lom.tick_params(labelsize=15)
                ax_rates_total_per_lom.grid('xy', linestyle=':', lw=0.5)
                fig_rates_total_per_lom.tight_layout()
                fig_rates_total_per_lom.savefig(os.path.join(output_dirs["coincidence_rates"], f"{lom_output_prefix}total_coincidence_rate_Th_{current_threshold_pC:.1f}pC_dT_{FIXED_COINCIDENCE_WINDOW_NS}ns.pdf"))
                plt.close(fig_rates_total_per_lom)

                # Log Scale version of total coincidence rate (individual LOMs)
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
                    ax_rates_total_log_per_lom.legend(fontsize=12)
                else:
                    ax_rates_total_log_per_lom.text(0.5, 0.5, "No positive rates to plot", horizontalalignment='center', verticalalignment='center', transform=ax_rates_total_log_per_lom.transAxes, fontsize=10, color='gray')
                
                ax_rates_total_log_per_lom.set_xlabel('PMT Channel ID',fontsize=12)
                ax_rates_total_log_per_lom.set_ylabel('Total Rate [1/s]',fontsize=12)
                ax_rates_total_log_per_lom.set_title(f"Total Coincidence Rate (Log Scale) - LOM {current_lom_name}\nTh: {current_threshold_pC:.1f} pC, dT: {FIXED_COINCIDENCE_WINDOW_NS}ns", fontsize=9)
                ax_rates_total_log_per_lom.tick_params(labelsize=15)
                ax_rates_total_log_per_lom.grid('xy', linestyle=':', lw=0.5)
                fig_rates_total_log_per_lom.tight_layout()
                fig_rates_total_log_per_lom.savefig(os.path.join(output_dirs["coincidence_rates_log"], f"{lom_output_prefix}total_coincidence_rate_log_Th_{current_threshold_pC:.1f}pC_dT_{FIXED_COINCIDENCE_WINDOW_NS}ns.pdf"))
                plt.close(fig_rates_total_log_per_lom)

        # --- End of Coincidence Analysis for current LOM ---

        # --- Start: FFT Analysis of 100 pC Peak Waveforms and Comparison Plots ---
        print("\n--- Starting FFT Analysis of 100 pC Peak Waveforms ---")
        # Define the peak charge range for waveform analysis
        min_charge_peak_pC = 80.0
        max_charge_peak_pC = 120.0

        #Small Signal
        # NEW: Define charge range for small signals
        min_charge_small_signal_pC = 0.0 # Example: adjust as needed
        max_charge_small_signal_pC = 20.0 # Example: adjust as needed

        current_min_charge = min_charge_peak_pC
        current_max_charge = max_charge_peak_pC
        charge_range_description = f"{min_charge_peak_pC}-{max_charge_peak_pC} pC (Bright Signal)"

            # Apply specific charge ranges based on PMT ID
        if pmt_id in [0, 1, 8, 9]:
                # PMT 00, 01, 08, 09: Bright Events (already set by default)
                pass
        else:
                # Other PMTs: Small Signal Events
                current_min_charge = min_charge_small_signal_pC
                current_max_charge = max_charge_small_signal_pC
                charge_range_description = f"{min_charge_small_signal_pC}-{max_charge_small_signal_pC} pC (Small Signal)"

        print(f"PMT {pmt_id}: Selecting events in charge range {charge_range_description}")

        timestamps_for_waveform_selection, event_indices_for_waveform_selection = get_timestamps(
                infile_pmt,
                min_charge_pC=current_min_charge,
                max_charge_pC=current_max_charge
            )

        if not timestamps_for_waveform_selection:
                print(f"PMT {pmt_id}: No events found in {charge_range_description} range. Skipping FFT and comparison for this PMT.")
                pmt_ch1_wf_data_for_comparison[pmt_id] = None
                pmt_ch1_fft_data_for_comparison[pmt_id] = None
                pmt_ch2_wf_data_for_comparison[pmt_id] = None
                pmt_ch2_fft_data_for_comparison[pmt_id] = None
                continue

            # Pick the first event in the selected range as representative
        representative_event_idx = event_indices_for_waveform_selection[0]
        representative_timestamp = timestamps_for_waveform_selection[0]

        rep_charges_ch1, rep_charges_ch2 = get_charges_of_these_events(infile_pmt, [representative_event_idx])
        rep_charge_ch1_pC = rep_charges_ch1[0] if rep_charges_ch1 else np.nan
        rep_charge_ch2_pC = rep_charges_ch2[0] if rep_charges_ch2 else np.nan

        print(f"PMT {pmt_id}: Analyzing representative Event {representative_event_idx} (Time: {representative_timestamp} ns, Ch1 Charge: {rep_charge_ch1_pC:.2f} pC, Ch2 Charge: {rep_charge_ch2_pC:.2f} pC) from selected range.")

        x_wf, wf1_data, wf2_data = get_waveform_at_this_timestamp(infile_pmt, representative_timestamp)
        
        pmt_ch1_wf_data_for_comparison = {}
        pmt_ch1_fft_data_for_comparison = {}
        pmt_ch2_wf_data_for_comparison = {}
        pmt_ch2_fft_data_for_comparison = {}

        for pmt_id in pmt_ids_from_map_sorted:
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

    # --- Finalize and save all combined Coincidence Matrices plots (across all LOMs) ---
    if matrices_linear_mappables:
        # Create a single colorbar for all linear matrices
        # Place it on the right side of the entire GridSpec
        cbar_ax_linear_matrices = fig_matrices_all_loms_linear.add_subplot(gs_matrices_linear[:, -1])
        fig_matrices_all_loms_linear.colorbar(matrices_linear_mappables[0], cax=cbar_ax_linear_matrices, 
                                                label='Coincidence Rate [Hz]', ticks=fixed_cbar_ticks)
    fig_matrices_all_loms_linear.tight_layout()
    matrices_linear_filename = os.path.join(output_dirs["coincidence_matrices"], f"All_LOMs_coincidence_matrices_linear_dT_{FIXED_COINCIDENCE_WINDOW_NS}ns.pdf")
    fig_matrices_all_loms_linear.savefig(matrices_linear_filename)
    plt.close(fig_matrices_all_loms_linear)
    print(f'Saved combined linear coincidence matrices for all LOMs to: {matrices_linear_filename}')

    if matrices_log_mappables:
        # Create a single colorbar for all log matrices
        # Place it on the right side of the entire GridSpec
        cbar_ax_log_matrices = fig_matrices_all_loms_log.add_subplot(gs_matrices_log[:, -1])
        fig_matrices_all_loms_log.colorbar(matrices_log_mappables[0], cax=cbar_ax_log_matrices, 
                                            label='Coincidence Rate [Hz] (Log Scale)', format='%.2e')
    fig_matrices_all_loms_log.tight_layout()
    matrices_log_filename = os.path.join(output_dirs["coincidence_matrices_log"], f"All_LOMs_coincidence_matrices_log_dT_{FIXED_COINCIDENCE_WINDOW_NS}ns.pdf")
    fig_matrices_all_loms_log.savefig(matrices_log_filename)
    plt.close(fig_matrices_all_loms_log)
    print(f'Saved combined log coincidence matrices for all LOMs to: {matrices_log_filename}')


    print("\n========================================================")
    print("=== All LOMs Processed. Analysis Complete. ===")
    print("========================================================")