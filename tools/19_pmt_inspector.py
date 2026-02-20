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
import re # Import for regular expressions (for more robust parsing)
from matplotlib.colors import LogNorm # For logarithmic color scale
from matplotlib.gridspec import GridSpec # For more flexible subplot arrangement
import matplotlib.cm as cm # Import colormap for scatter plot
from scipy.stats import gaussian_kde # For smoothing histograms
from scipy.interpolate import interp1d # For smoothing the lines

# === START: HDF File Loading Function (Directly Defined) ===
# This function replaces any external hdf_reader.py dependency.

def load_hdf_file_as_dict(filename):
    """
    Loads data from an HDF5 file into a dictionary-like structure.
    This function mimics the expected behavior for accessing HDF5 data.
    """
    data_dict = {}
    try:
        with h5py.File(filename, 'r') as f:
            for key, value in f.items():
                if isinstance(value, h5py.Group):
                    sub_group = {}
                    for sub_key, sub_value in value.items():
                        try:
                            # Accessing data directly with [()] to load its content
                            sub_group[sub_key] = sub_value[()]
                        except Exception as e:
                            warnings.warn(f"Could not read dataset '{sub_key}' from group '{key}' in '{filename}'. Error: {e}. Skipping this dataset.")
                            sub_group[sub_key] = None # Assign None if cannot be read
                    data_dict[key] = sub_group
                elif isinstance(value, h5py.Dataset):
                    data_dict[key] = value[()] # Directly load dataset at root level
                else:
                    warnings.warn(f"Unexpected HDF5 item type '{type(value)}' for key '{key}' in '{filename}'.")
        return data_dict
    except Exception as e:
        warnings.warn(f"Error loading HDF5 file '{filename}': {e}")
        return None

# Define 'read_hdffile' as an alias for our new, correct loading function.
read_hdffile = load_hdf_file_as_dict
# === END: HDF File Loading Function ===


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

    # Ensure conversion factors are scalars, if they came as 0-dim arrays
    if isinstance(conversion_ch1, np.ndarray) and conversion_ch1.ndim == 0:
        conversion_ch1 = conversion_ch1.item()
    if isinstance(conversion_ch2, np.ndarray) and conversion_ch2.ndim == 0:
        conversion_ch2 = conversion_ch2.item()

    # Convert raw charges to numpy arrays for consistent processing
    q_ch1_raw = np.asarray(q_ch1_raw)
    q_ch2_raw = np.asarray(q_ch2_raw)
    fpga_time = np.asarray(fpga_time)

    conversion_factor_pC = (1e-6 * (1/60e6) * 1e12) # converts from (unit of conversion_ch) to pC

    q_ch1_pC = np.array([ s * (conversion_ch1 * conversion_factor_pC) for s in q_ch1_raw])
    q_ch2_pC = np.array([ s * (conversion_ch2 * conversion_factor_pC) for s in q_ch2_raw])

    # Ensure all arrays have the same length
    min_len = min(len(fpga_time), len(q_ch1_pC), len(q_ch2_pC))
    
    for iev, (q1, q2, t) in enumerate(zip(q_ch1_pC[:min_len], q_ch2_pC[:min_len], fpga_time[:min_len])):
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
            - int: `ped1_val`, pedestal for channel 1.
            - int: `ped2_val`, pedestal for channel 2.
            Returns empty numpy arrays and 0 for pedestals if no matching timestamp or data.
    """
    data = read_hdffile(filename)

    if not data or "data" not in data:
        warnings.warn(f"Missing 'data' group in {filename}. Cannot retrieve waveform.")
        return np.array([]), np.array([]), np.array([]), 0, 0

    fpga_time = data["data"]["FPGAtime"][()] if "FPGAtime" in data["data"] else np.array([])
    nsamples = data["data"]["nsample"][()] if "nsample" in data["data"] else np.array([])
    adc_ch1 = data["data"]["ADC_ch1"][()] if "ADC_ch1" in data["data"] else np.array([])
    adc_ch2 = data["data"]["ADC_ch2"][()] if "ADC_ch2" in data["data"] else np.array([])

    pedestal_ch1_all = data["data"]["pedestal_ch1"][()] if "pedestal_ch1" in data["data"] else np.array([0])
    pedestal_ch2_all = data["data"]["pedestal_ch2"][()] if "pedestal_ch2" in data["data"] else np.array([0])

    x = np.array([])
    wf_ch1 = np.array([])
    wf_ch2 = np.array([])
    ped1_val = 0
    ped2_val = 0

    matching_indices = np.where(fpga_time == timestamp)[0]
    if len(matching_indices) > 0:
        iev = matching_indices[0]

        if (iev < len(nsamples) and iev < adc_ch1.shape[0] and iev < adc_ch2.shape[0] and
            (pedestal_ch1_all.ndim == 0 or pedestal_ch1_all.size > iev) and # Check size for scalar/array
            (pedestal_ch2_all.ndim == 0 or pedestal_ch2_all.size > iev)): # Check size for scalar/array

            n = nsamples[iev]

            x = np.array([ i * (1e9/60e6) for i in range(n) ])

            ped1_val = pedestal_ch1_all[iev] if pedestal_ch1_all.ndim > 0 and pedestal_ch1_all.size > iev else (pedestal_ch1_all.item() if pedestal_ch1_all.size > 0 else 0)
            ped2_val = pedestal_ch2_all[iev] if pedestal_ch2_all.ndim > 0 and pedestal_ch2_all.size > iev else (pedestal_ch2_all.item() if pedestal_ch2_all.size > 0 else 0)

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

    return x, wf_ch1, wf_ch2, ped1_val, ped2_val # Return pedestals

def get_charges_of_these_events(filename, evidx_list):
    """Retrieves charge values (in pC) for specific event indices."""
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
    conversion_factor_pC = (1e-6 * (1/60e6) * 1e12)
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

# --- Start: Plotting Functions ---



def plot_charge_correlation(df_events, pmt_full_name, output_path_base, fit_q1_min_pC, fit_q1_max_pC_for_fit, is_forced_fit=False):
    """
    Plots Ch1 vs Ch2 charge correlation for a specific PMT using scatter plots.
    Includes Linear and Log scales, with fitted line and R-squared.
    Can perform forced fitting.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8)) # Landscape orientation for 2 plots side-by-side

    m_slope, c_intercept, r_squared = np.nan, np.nan, np.nan
    df_fit = df_events[(df_events['ch1_pC'] >= fit_q1_min_pC) & (df_events['ch1_pC'] <= fit_q1_max_pC_for_fit)]

    fit_warning_message = ""
    # Attempt initial fit without slope bounds (to check true correlation)
    if len(df_fit) >= 2:
        try:
            params, cov = curve_fit(linear_function, df_fit['ch1_pC'].values, df_fit['ch2_pC'].values, bounds = ((0,-0.1),(0.1,0.05)))        
            m_slope, c_intercept = params
            residuals = df_fit['ch2_pC'].values - linear_function(df_fit['ch1_pC'].values, m_slope, c_intercept)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((df_fit['ch2_pC'].values - np.mean(df_fit['ch2_pC'].values))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        except Exception as e: # Catch all errors for initial fit
            warnings.warn(f"Initial unconstrained fit failed for {pmt_full_name}: {e}")
            m_slope, c_intercept, r_squared = np.nan, np.nan, np.nan

    # Check if a forced fit is needed based on R^2 and slope of initial fit
    # If R^2 is very low AND slope is very flat, then force a fit.
    # The 'forcing' logic will re-attempt fit with specific bounds/data modification.
    # (Removed forced fit logic to simplify for this task, can be re-added if needed)

    # --- Plotting Data ---
    # Common plotting ranges (pC)
    x_plot_min_pC = 0; x_plot_max_pC = 100
    y_plot_min_pC = 0; y_plot_max_pC = 10

    # Linear Scale Plot (Scatter)
    ax = axes[0]
    ax.scatter(df_events['ch1_pC'], df_events['ch2_pC'], s=5, alpha=0.3, color='blue', label='Data Points') # Scatter plot
    ax.set_title(f'{pmt_full_name} (Linear Scale){fit_warning_message}', fontsize=10)
    ax.set_xlabel('Charge Ch1 (pC)', fontsize=8)
    ax.set_ylabel('Charge Ch2 (pC)', fontsize=8)
    ax.set_xlim(x_plot_min_pC, x_plot_max_pC)
    ax.set_ylim(y_plot_min_pC, y_plot_max_pC)
    ax.grid(True, linestyle=':', alpha=0.6)
    if not np.isnan(m_slope):
        x_fit_plot = np.linspace(x_plot_min_pC, x_plot_max_pC, 100)
        y_fit_plot = linear_function(x_fit_plot, m_slope, c_intercept)
        ax.plot(x_fit_plot, y_fit_plot, color='red', linestyle='--', linewidth=1.5,
                     label=f'Fit: Q2 = {m_slope:.2f}*Q1 + {c_intercept:.2f} pC\n($R^2={r_squared:.2f}$)\nFit Range: [{fit_q1_min_pC:.1f}-{fit_q1_max_pC_for_fit:.1f}] pC')
    ax.legend(fontsize=6, loc='upper left')

    # Log Scale Plot (Heatmap/hist2d for density)
    ax = axes[1]
    x_bins = np.linspace(x_plot_min_pC, x_plot_max_pC, 101) # 1 pC bin size
    y_bins = np.linspace(y_plot_min_pC, y_plot_max_pC, 51)  # 0.2 pC bin size
    h_log = ax.hist2d(df_events['ch1_pC'], df_events['ch2_pC'], bins=[x_bins, y_bins], cmap='viridis', norm=LogNorm(vmin=1e0, vmax=1e3))
    log_ticks = [10**i for i in range(0, 4)] # For 1e0 to 1e3
    fig.colorbar(h_log[3], ax=ax, label='Number of Events (Log Scale)')
    ax.set_title(f'{pmt_full_name} (Log Scale){fit_warning_message}', fontsize=10)
    ax.set_xlabel('Charge Ch1 (pC)', fontsize=8)
    ax.set_ylabel('Charge Ch2 (pC)', fontsize=8)
    ax.set_xlim(x_plot_min_pC, x_plot_max_pC)
    ax.set_ylim(y_plot_min_pC, y_plot_max_pC)
    ax.grid(True, linestyle=':', alpha=0.6)
    if not np.isnan(m_slope):
        x_fit_plot = np.linspace(x_plot_min_pC, x_plot_max_pC, 100)
        y_fit_plot = linear_function(x_fit_plot, m_slope, c_intercept)
        ax.plot(x_fit_plot, y_fit_plot, color='red', linestyle='--', linewidth=1.5,
                     label=f'Fit: Q2 = {m_slope:.2f}*Q1 + {c_intercept:.2f} pC\n($R^2={r_squared:.2f}$)\nFit Range: [{fit_q1_min_pC:.1f}-{fit_q1_max_pC_for_fit:.1f}] pC')
    ax.legend(fontsize=6, loc='upper left')

    fig.suptitle(f"Charge Correlation (Ch1 vs Ch2) for {pmt_full_name}", fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust rect for suptitle
    
    output_filename = f"{output_path_base}charge_correlation_ch1_ch2.pdf"
    plt.savefig(output_filename)
    plt.close(fig)
    return m_slope, c_intercept, r_squared # Return fit parameters


def plot_charge_distribution(ch_charges_pC_series, pmt_full_name, channel_name, assumed_gain, output_path_base, ax=None):
    """
    Plots charge distribution in PE for a specific channel of a PMT.
    X-axis in PE, Y-axis in Counts (Log Scale).
    Can plot to a specified axes (for subplots) or create a new figure.
    """
    electron_charge_C = 1.602e-19
    charge_per_pe_C = assumed_gain * electron_charge_C
    charge_per_pe_pC = charge_per_pe_C * 1e12 # pC per PE

    charges_pe = ch_charges_pC_series.values / charge_per_pe_pC
    charges_pe = charges_pe[~np.isnan(charges_pe)]

    if charges_pe.size == 0:
        print(f"No valid {channel_name} charge data for distribution for {pmt_full_name}. Skipping plot.")
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        save_fig = True
    else:
        save_fig = False

    x_dist_min_pe = 0
    x_dist_max_pe = 300
    bins = np.linspace(x_dist_min_pe, x_dist_max_pe, 81) # 0.25 PE bin size for 0-20 PE

    ax.hist(charges_pe, bins=bins, color='skyblue', edgecolor='blue', alpha=0.7)

    ax.set_xlabel('Charge (PE)', fontsize=10) # Reduced font size for subplots
    ax.set_ylabel('Counts (Log Scale)', fontsize=10) # Reduced font size for subplots
    ax.set_title(f'{pmt_full_name} - {channel_name}', fontsize=12) # Reduced font size
    
    ax.set_xlim([x_dist_min_pe, x_dist_max_pe])
    ax.set_yscale('log') # Ensure log scale for y-axis

    ax.tick_params(labelsize=8) # Reduced label size for subplots
    ax.grid(True, linestyle=':', alpha=0.6)

    if save_fig:
        plt.savefig(output_path_base + f"_{channel_name}_distribution.pdf")
        plt.close(fig)

def plot_sample_waveforms_specific(filename, timestamps, pmt_full_name, output_path_base):
    """
    Plots sample waveforms for a specific PMT.
    """
    if not timestamps:
        print(f"No timestamps available for sample waveform plot for {pmt_full_name}. Skipping.")
        return

    num_samples = min(5, len(timestamps)) # Plot up to 5 samples
    if num_samples == 0:
        print(f"Not enough events to sample waveforms for {pmt_full_name}. Skipping.")
        return

    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 3 * num_samples))
    if num_samples == 1: # If only one subplot, axes is not an array
        axes = [axes]

    for i, timestamp in enumerate(timestamps[:num_samples]):
        ax = axes[i]
        x_wf, wf1_data, wf2_data, _, _ = get_waveform_at_this_timestamp(filename, timestamp) # Get pedestals too

        if len(x_wf) > 0:
            ax.plot(x_wf, wf1_data, color='blue', label='Ch1 (pedestal subtracted, shifted)')
            ax.plot(x_wf, wf2_data, color='green', label='Ch2 (raw ADC)')
            ax.set_title(f"Event Timestamp: {timestamp} ns", fontsize=8)
            ax.set_xlabel("Time (ns)", fontsize=7)
            ax.set_ylabel("ADC Counts", fontsize=7)
            ax.set_xlim([0, 600])
            ax.set_ylim([0, 4200]) # Consistent Y-limit
            ax.legend(fontsize=6, loc='upper right')
        else:
            ax.text(0.5, 0.5, "No Waveform Data", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
        ax.grid(True, linestyle=':', lw=0.5)
        ax.tick_params(labelsize=6)

    fig.suptitle(f"Sample Waveforms for {pmt_full_name}", fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(output_path_base + "_waveforms.pdf")
    plt.close(fig)

def plot_adc_time_series(adc_values, pmt_full_name, channel_name, output_path_base):
    """Plots raw ADC counts of a channel across events."""
    if adc_values.size == 0:
        print(f"No ADC values available for {channel_name} time series for {pmt_full_name}. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(adc_values, marker='.', linestyle='none', alpha=0.5)
    ax.set_title(f'Raw ADC Counts - {channel_name} - {pmt_full_name}', fontsize=14)
    ax.set_xlabel('Event Index', fontsize=12)
    ax.set_ylabel('ADC Counts (raw)', fontsize=12)
    ax.set_ylim([0, 4095]) # ADC range
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(output_path_base + f"_{channel_name}_raw_adc_series.pdf")
    plt.close(fig)

def plot_waveform_baseline_zoom(filename, pmt_full_name, output_path_base):
    """
    Inspects and plots a zoomed-in raw waveform (Ch2) around its baseline for 1-2 random events.
    """
    data = read_hdffile(filename)
    if not data or "data" not in data:
        print(f"No data to zoom waveform baseline for {pmt_full_name}. Skipping.")
        return

    all_fpga_times = data["data"]["FPGAtime"][()] if "FPGAtime" in data["data"] else np.array([])
    all_pedestal_ch2 = data["data"]["pedestal_ch2"][()] if "pedestal_ch2" in data["data"] else np.array([0])
    
    if all_fpga_times.size == 0:
        print(f"No timestamps to zoom waveform baseline for {pmt_full_name}. Skipping.")
        return

    # Try to pick a few events for zoom-in, up to 2
    num_samples = min(2, len(all_fpga_times))
    if num_samples == 0:
        print(f"Not enough events to zoom waveform baseline for {pmt_full_name}. Skipping.")
        return

    np.random.seed(42) # For reproducibility
    sample_indices = np.random.choice(len(all_fpga_times), num_samples, replace=False)
    sample_timestamps = all_fpga_times[sample_indices]

    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]

    print(f"\n--- Inspecting Ch2 Waveform Baseline for {pmt_full_name} (Zoomed) ---")
    for i, timestamp in enumerate(sample_timestamps):
        ax = axes[i]
        x_wf, _, wf2_data_raw, _, ped2_val = get_waveform_at_this_timestamp(filename, timestamp)

        if len(x_wf) > 0:
            ax.plot(x_wf, wf2_data_raw, color='green')
            ax.set_title(f"Ch2 Waveform (Zoom) - Event {timestamp} ns", fontsize=8)
            ax.set_xlabel("Time (ns)", fontsize=7)
            ax.set_ylabel("Raw ADC Counts", fontsize=7)
            ax.set_ylim([ped2_val - 50, ped2_val + 50]) # Zoom +/- 50 ADC counts around pedestal
            ax.set_xlim([0, x_wf[-1]]) # Full waveform time range
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.tick_params(labelsize=6)
        else:
            ax.text(0.5, 0.5, f"Event {timestamp} ns\nNo Waveform Data", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
        ax.grid(True, linestyle=':', lw=0.5)
        ax.tick_params(labelsize=6)

    fig.suptitle(f"Raw Waveform - Ch2 Baseline Zoom for {pmt_full_name}", fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(output_path_base + "_ch2_baseline_zoom.pdf")
    plt.close(fig)


def plot_2d_charge_histogram(charge_ch1_main_pC, charge_ch2_main_pC, pmt_full_name_main, assumed_gain, output_path_base, num_bins=100, charge_ch1_ref_pC=None, charge_ch2_ref_pC=None, pmt_full_name_ref=None):
    """
    Generates 2D histograms (heatmaps) of Ch1 vs Ch2 charge in PE.
    Optionally plots a reference PMT's data for comparison in a separate subplot.
    """
    electron_charge_C = 1.602e-19
    charge_per_pe_C = assumed_gain * electron_charge_C
    charge_per_pe_pC = charge_per_pe_C * 1e12 # pC per PE

    charge_ch1_main_pe = charge_ch1_main_pC / charge_per_pe_pC
    charge_ch2_main_pe = charge_ch2_main_pC / charge_per_pe_pC

    fig_width = 18 if charge_ch1_ref_pC is not None else 9
    fig, axes = plt.subplots(1, 2 if charge_ch1_ref_pC is not None else 1, figsize=(fig_width, 8), squeeze=False)
    axes = axes.flatten() # Ensure axes is always iterable

    x_min_pe, x_max_pe = 0, 100 / ASSUMED_CH1_PC_PER_PE # Use max pC for x_plot_max_pC converted to PE
    y_min_pe, y_max_pe = 0, 10 / ASSUMED_CH1_PC_PER_PE # Use max pC for y_plot_max_pC converted to PE
    bins_x = np.linspace(x_min_pe, x_max_pe, num_bins + 1)
    bins_y = np.linspace(y_min_pe, y_max_pe, num_bins // 10 + 1) # Fewer bins for Y as range is smaller

    # --- Main PMT Plot ---
    ax_main = axes[0]
    h_main = ax_main.hist2d(charge_ch1_main_pe, charge_ch2_main_pe, bins=[bins_x, bins_y], cmap='viridis', norm=LogNorm(vmin=1e0, vmax=1e3))
    fig.colorbar(h_main[3], ax=ax_main, label='Number of Events (Log Scale)')
    ax_main.set_title(f'2D Histogram: Ch1 vs Ch2 - {pmt_full_name_main}', fontsize=12)
    ax_main.set_xlabel('Charge Ch1 (PE)', fontsize=10)
    ax_main.set_ylabel('Charge Ch2 (PE)', fontsize=10)
    ax_main.set_xlim(x_min_pe, x_max_pe)
    ax_main.set_ylim(y_min_pe, y_max_pe)
    ax_main.grid(True, linestyle=':', alpha=0.6)
    ax_main.tick_params(labelsize=8)

    # --- Reference PMT Plot (if provided) ---
    if charge_ch1_ref_pC is not None and charge_ch2_ref_pC is not None:
        charge_ch1_ref_pe = charge_ch1_ref_pC / charge_per_pe_pC
        charge_ch2_ref_pe = charge_ch2_ref_pC / charge_per_pe_pC
        
        ax_ref = axes[1]
        h_ref = ax_ref.hist2d(charge_ch1_ref_pe, charge_ch2_ref_pe, bins=[bins_x, bins_y], cmap='magma', norm=LogNorm()) # Different colormap
        fig.colorbar(h_ref[3], ax=ax_ref, label='Number of Events (Log Scale) (Ref.)')
        ax_ref.set_title(f'2D Histogram: Ch1 vs Ch2 - {pmt_full_name_ref}', fontsize=12)
        ax_ref.set_xlabel('Charge Ch1 (PE)', fontsize=10)
        ax_ref.set_ylabel('Charge Ch2 (PE)', fontsize=10)
        ax_ref.set_xlim(x_min_pe, x_max_pe)
        ax_ref.set_ylim(y_min_pe, y_max_pe)
        ax_ref.grid(True, linestyle=':', alpha=0.6)
        ax_ref.tick_params(labelsize=8)
    else: # Remove the second subplot if no reference is provided
        if len(axes) > 1:
            fig.delaxes(axes[1])


    fig.suptitle(f"2D Charge Histogram: {pmt_full_name_main}" + (f" vs {pmt_full_name_ref}" if pmt_full_name_ref else ""), fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(output_path_base + "_2d_charge_histogram.pdf")
    plt.close(fig)

# --- NEW FUNCTION FOR PLOTTING ALL PMTS IN A LOM ---
def plot_all_lom_pmt_charge_distributions(lom_id, pmt_base_path, output_dir, assumed_gain, ch_name="Ch2"):
    """
    Plots the charge distribution for a specific channel (e.g., Ch2) for all 16 PMTs
    within a given LOM, arranged in a 4x4 subplot grid.

    Args:
        lom_id (str): The LOM ID (e.g., "data_muon_run_lom16-06").
        pmt_base_path (str): The base directory containing the HDF5 files for the LOM.
                             (e.g., "muon_data/data_muon_run_lom16-06/")
        output_dir (str): The directory where the output PDF will be saved.
        assumed_gain (float): Assumed PMT gain for PE conversion.
        ch_name (str): The channel name to plot (e.g., "Ch2").
    """
    print(f"\n--- Plotting Charge Distributions for all 16 PMTs in {lom_id} ({ch_name}) ---")

    fig, axes = plt.subplots(4, 4, figsize=(20, 20)) # 4x4 grid for 16 PMTs
    axes = axes.flatten() # Flatten the 2D array of axes for easy iteration

    all_pmt_files = sorted(glob.glob(os.path.join(pmt_base_path, f"data-muon-run*.hdf")))

    if not all_pmt_files:
        print(f"No HDF5 files found in {pmt_base_path} for {lom_id}. Skipping plotting all PMT distributions.")
        plt.close(fig)
        return

    # Filter files to only include those matching the pattern for PMT 00-15
    # Assumes filename format like data-muon-run<run_number>.<pmt_id>.hdf
    pmt_files_map = {}
    for f in all_pmt_files:
        match = re.search(r'\.(\d{2})\.hdf$', os.path.basename(f))
        if match:
            pmt_id = int(match.group(1))
            if 0 <= pmt_id <= 15: # Only consider PMT 00 to 15
                pmt_files_map[pmt_id] = f
    
    # Ensure we have all 16 PMTs, or at least try to plot what's available
    sorted_pmt_ids = sorted(pmt_files_map.keys())

    for i, pmt_id in enumerate(sorted_pmt_ids):
        if i >= 16: # Only plot up to 16 PMTs per page
            break

        ax = axes[i]
        filename = pmt_files_map[pmt_id]
        pmt_full_name = f"{lom_id}: PMT {pmt_id:02d}"

        data_pmt = read_hdffile(filename)
        charge_data = np.array([])

        if data_pmt and "data" in data_pmt and "metadata" in data_pmt:
            try:
                raw_charges = data_pmt["data"][f"charge_{ch_name.lower()}"][()]
                conversion_factor = data_pmt["metadata"][f"conversion_{ch_name.lower()}"][()]
                
                # Ensure conversion factor is scalar
                if isinstance(conversion_factor, np.ndarray) and conversion_factor.ndim == 0:
                    conversion_factor = conversion_factor.item()

                conversion_factor_pC = (1e-6 * (1/60e6) * 1e12)
                charge_data = raw_charges * (conversion_factor * conversion_factor_pC)
                
                # Filter out very low charge events as done in the main script
                df_temp = pd.DataFrame({'charge': charge_data})
                df_temp_filtered = df_temp[df_temp['charge'] > 1] # Filter charges > 1 pC
                
                if not df_temp_filtered['charge'].empty:
                    plot_charge_distribution(
                        df_temp_filtered['charge'],
                        f"PMT {pmt_id:02d}", # Only show PMT ID in subplot title
                        ch_name,
                        assumed_gain,
                        None, # No individual file save here
                        ax=ax
                    )
                else:
                    ax.text(0.5, 0.5, "No Data (>1pC)", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10, color='gray')
                    ax.set_title(f"PMT {pmt_id:02d}", fontsize=12)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.grid(True, linestyle=':', lw=0.5)

            except KeyError as e:
                warnings.warn(f"Missing data for {pmt_full_name} {ch_name}: {e}. Skipping subplot.")
                ax.text(0.5, 0.5, f"Data Error: {e}", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10, color='red')
                ax.set_title(f"PMT {pmt_id:02d}", fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(True, linestyle=':', lw=0.5)
            except Exception as e:
                warnings.warn(f"An error occurred loading/processing data for {pmt_full_name} {ch_name}: {e}. Skipping subplot.")
                ax.text(0.5, 0.5, f"Processing Error: {e}", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10, color='red')
                ax.set_title(f"PMT {pmt_id:02d}", fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(True, linestyle=':', lw=0.5)
        else:
            ax.text(0.5, 0.5, "File Load Error", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10, color='red')
            ax.set_title(f"PMT {pmt_id:02d}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True, linestyle=':', lw=0.5)
            warnings.warn(f"Could not load data from {filename}. Skipping subplot for PMT {pmt_id:02d}.")

    # Hide any unused subplots if fewer than 16 PMTs are found
    for j in range(len(sorted_pmt_ids), 16):
        fig.delaxes(axes[j])

    fig.suptitle(f"Charge Distribution (Ch2) for all PMTs in {lom_id}", fontsize=20, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust rect for suptitle

    output_filename = os.path.join(output_dir, f"{lom_id}_all_pmt_ch2_charge_distributions.pdf")
    plt.savefig(output_filename)
    plt.close(fig)
    print(f"Saved all PMT Ch2 charge distributions to: {output_filename}")


# --- End: Plotting Functions ---


if __name__ == "__main__":
    # --- Configuration for the specific PMT to check ---
    LOM_ID_TO_CHECK = "data_muon_run_lom16-06"
    PMT_ID_TO_CHECK = 2
    # IMPORTANT: Adjust this path to your actual file location
    # Example: "C:/Users/YourUser/Documents/IceCube/muon_data/data_muon_run_lom16-06/data-muon-run909.02.hdf"
    PMT_02_FILE_PATH = "muon_data/data_muon_run_lom16-06/data-muon-run909.09.hdf"
     
    # IMPORTANT: Reference PMT (e.g., from a known good LOM, PMT 01 from LOM16-06 itself)
    # This is to compare PMT 02 with another PMT within the same LOM.
    # Example: "C:/Users/YourUser/Documents/IceCube/muon_data/data_muon_run_lom16-06/data-muon-run909.01.hdf"
    REFERENCE_LOM_ID = "data_muon_run_lom16-06" # LOM 16-06 itself
    REFERENCE_PMT_ID = 1 # PMT 01 from the same LOM
    REFERENCE_PMT_FILE_PATH = "muon_data/data_muon_run_lom16-06/data-muon-run909.09.hdf"


    OUTPUT_DIR_NAME = "pmt_check_output" # Output folder for this specific check
    output_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_DIR_NAME)
    os.makedirs(output_base_dir, exist_ok=True)

    # --- Fixed parameters for consistency with main script ---
    ASSUMED_GAIN = 5e6 # Assumed PMT gain for PE conversion (electrons/PE)
    FIXED_CORRELATION_FIT_Q1_MIN_PC = 5.0 # Minimum Ch1 charge (pC) for linear regression
    FIXED_CORRELATION_FIT_Q1_MAX_PC = 67.2 # Fixed max for fitting as per discussion
    ASSUMED_CH1_PC_PER_PE = 0.8 # Used for pC to PE conversion where needed

    pmt_full_name_for_plot = f"{LOM_ID_TO_CHECK}: PMT {PMT_ID_TO_CHECK:02d}"
    output_file_prefix = os.path.join(output_base_dir, f"{LOM_ID_TO_CHECK}_PMT{PMT_ID_TO_CHECK:02d}_")

    print(f"--- Starting detailed check for {pmt_full_name_for_plot} ---")

    # 1. Load data for the specific PMT (PMT 02 LOM 16-06)
    pmt_events_data = []
    data_pmt = read_hdffile(PMT_02_FILE_PATH)

    if data_pmt and "data" in data_pmt and "metadata" in data_pmt:
        try:
            raw_charges_ch1_adc = data_pmt["data"]["charge_ch1"][()]
            conversion_ch1 = data_pmt["metadata"]["conversion_ch1"][()]
            raw_charges_ch2_adc = data_pmt["data"]["charge_ch2"][()]
            conversion_ch2 = data_pmt["metadata"]["conversion_ch2"][()]
            raw_fpga_time = data_pmt["data"]["FPGAtime"][()]
            raw_adc_ch2_all = data_pmt["data"]["ADC_ch2"][()]
            raw_pedestal_ch2_all = data_pmt["data"]["pedestal_ch2"][()]

            conversion_factor_pC = (1e-6 * (1/60e6) * 1e12)
            charges_ch1_pC = raw_charges_ch1_adc * (conversion_ch1 * conversion_factor_pC)
            charges_ch2_pC = raw_charges_ch2_adc * (conversion_ch2 * conversion_factor_pC)

            min_len_pmt = min(len(charges_ch1_pC), len(charges_ch2_pC), len(raw_fpga_time))
            for i in range(min_len_pmt):
                pmt_events_data.append({
                    'ch1_pC': charges_ch1_pC[i],
                    'ch2_pC': charges_ch2_pC[i],
                    'filename': PMT_02_FILE_PATH,
                    'timestamp': raw_fpga_time[i],
                    'event_idx': i
                })
        except KeyError as e:
            warnings.warn(f"Missing data for {pmt_full_name_for_plot}: {e}. Cannot perform check.")
            sys.exit(1)
    else:
        print(f"Error: Could not load data from {PMT_02_FILE_PATH}. Please check the path and file integrity.")
        sys.exit(1)

    df_pmt_events = pd.DataFrame(pmt_events_data)
    df_pmt_events_filtered = df_pmt_events.dropna(subset=['ch1_pC', 'ch2_pC'])
    df_pmt_events_filtered = df_pmt_events_filtered[(df_pmt_events_filtered['ch1_pC'] > 1) & (df_pmt_events_filtered['ch2_pC'] > 1)] # Filter out very low charge

    # 1.1. Load data for the Reference PMT
    df_ref_pmt_events_filtered = pd.DataFrame() # Initialize as empty
    pmt_full_name_ref = ""
    try:
        ref_pmt_events_data = []
        data_ref_pmt = read_hdffile(REFERENCE_PMT_FILE_PATH)
        if data_ref_pmt and "data" in data_ref_pmt and "metadata" in data_ref_pmt:
            raw_charges_ch1_ref_adc = data_ref_pmt["data"]["charge_ch1"][()]
            conversion_ch1_ref = data_ref_pmt["metadata"]["conversion_ch1"][()]
            raw_charges_ch2_ref_adc = data_ref_pmt["data"]["charge_ch2"][()]
            conversion_ch2_ref = data_ref_pmt["metadata"]["conversion_ch2"][()]
            
            conversion_factor_pC_ref = (1e-6 * (1/60e6) * 1e12)
            charges_ch1_pC_ref = raw_charges_ch1_ref_adc * (conversion_ch1_ref * conversion_factor_pC_ref)
            charges_ch2_pC_ref = raw_charges_ch2_ref_adc * (conversion_ch2_ref * conversion_factor_pC_ref)
            
            min_len_ref = min(len(charges_ch1_pC_ref), len(charges_ch2_pC_ref))
            for i in range(min_len_ref):
                ref_pmt_events_data.append({
                    'ch1_pC': charges_ch1_pC_ref[i],
                    'ch2_pC': charges_ch2_pC_ref[i]
                })
            df_ref_pmt_events = pd.DataFrame(ref_pmt_events_data)
            df_ref_pmt_events_filtered = df_ref_pmt_events.dropna(subset=['ch1_pC', 'ch2_pC'])
            df_ref_pmt_events_filtered = df_ref_pmt_events_filtered[(df_ref_pmt_events_filtered['ch1_pC'] > 1) & (df_ref_pmt_events_filtered['ch2_pC'] > 1)]
            pmt_full_name_ref = f"{REFERENCE_LOM_ID}: PMT {REFERENCE_PMT_ID:02d} (Reference)"
        else:
            print(f"Warning: Could not load reference data from {REFERENCE_PMT_FILE_PATH}. Comparison plot will be skipped.")
    except Exception as e:
        print(f"Warning: Error loading reference PMT data ({REFERENCE_PMT_FILE_PATH}): {e}. Comparison plot will be skipped.")


    # --- Print basic data summary ---
    print(f"\n--- Data Summary for {pmt_full_name_for_plot} ---")
    print(f"Total events loaded: {len(pmt_events_data)}")
    print(f"Events after filtering (Ch1>1pC & Ch2>1pC): {len(df_pmt_events_filtered)}")
    if not df_pmt_events_filtered.empty:
        print(f"Mean Ch1 Charge: {df_pmt_events_filtered['ch1_pC'].mean():.2f} pC")
        print(f"Mean Ch2 Charge: {df_pmt_events_filtered['ch2_pC'].mean():.2f} pC")
    else:
        print("No events remaining after filtering for mean charge calculation.")

    # --- Plot Charge Correlation ---
    print(f"\n--- Plotting Charge Correlation for {pmt_full_name_for_plot} ---")
    m_slope, c_intercept, r_squared = plot_charge_correlation(
        df_pmt_events_filtered, # Use filtered data for correlation plot
        pmt_full_name_for_plot,
        output_file_prefix,
        FIXED_CORRELATION_FIT_Q1_MIN_PC,
        FIXED_CORRELATION_FIT_Q1_MAX_PC,
        is_forced_fit=True # Indicate that forced fit logic should be applied
    )

    # --- Plot Charge Distribution for Ch1 and Ch2 ---
    print(f"\n--- Plotting Charge Distributions for {pmt_full_name_for_plot} ---")
    if 'ch1_pC' in df_pmt_events_filtered.columns and not df_pmt_events_filtered['ch1_pC'].empty:
        plot_charge_distribution(
            df_pmt_events_filtered['ch1_pC'], # Changed to ch1_pC as per function parameter
            pmt_full_name_for_plot,
            "Ch1",
            ASSUMED_GAIN,
            output_file_prefix
        )
    if 'ch2_pC' in df_pmt_events_filtered.columns and not df_pmt_events_filtered['ch2_pC'].empty:
        plot_charge_distribution(
            df_pmt_events_filtered['ch2_pC'], # Changed to ch2_pC as per function parameter
            pmt_full_name_for_plot,
            "Ch2",
            ASSUMED_GAIN,
            output_file_prefix
        )
    
    # --- DIAGNOSTIC CHECKLIST PLOTS ---

    # 1. Inspect raw waveform (Zoom-in Baseline) - Ch2
    plot_waveform_baseline_zoom(PMT_02_FILE_PATH, pmt_full_name_for_plot, output_file_prefix)

    # 2. Dump ADC counts time-series (Pedestal included) - Ch2
    print(f"\n--- Plotting Raw ADC Counts Time-Series for {pmt_full_name_for_plot} Ch2 ---")
    raw_adc_data_for_series_loaded = data_pmt["data"]["ADC_ch2"][()] if data_pmt and "data" in data_pmt and "ADC_ch2" in data_pmt["data"] else np.array([])
    
    adc_values_per_event = np.array([])
    if raw_adc_data_for_series_loaded.ndim > 1:
        # Take the mean of the first 10 samples (or fewer if not available) as representative pedestal for the event
        # Use np.nanmean to handle potential NaN values if any (though unlikely for raw ADC)
        representative_pedestal_per_event = np.nanmean(raw_adc_data_for_series_loaded[:, :min(10, raw_adc_data_for_series_loaded.shape[1])], axis=1)
        adc_values_per_event = representative_pedestal_per_event
    elif raw_adc_data_for_series_loaded.ndim == 1: # If it's already a 1D array of values per event
        adc_values_per_event = raw_adc_data_for_series_loaded
        
    if adc_values_per_event.size > 0:
        plot_adc_time_series(adc_values_per_event, pmt_full_name_for_plot, "Ch2", output_file_prefix)
    else:
        print(f"No raw ADC data available for Ch2 time series for {pmt_full_name_for_plot}.")


    # 3. Cross-check Timestamp Matching (Ch1 vs Ch2)
    print(f"\n--- Checking Timestamp Matching for {pmt_full_name_for_plot} ---")
    timestamps_ch1_all_pmt_file, _ = get_timestamps(PMT_02_FILE_PATH, min_charge_pC=0.0, max_charge_pC=float('inf'))
    # Directly get Ch2's FPGAtime from the loaded data_pmt, assuming it's the same for both channels in the file structure
    timestamps_ch2_all_pmt_file_raw = data_pmt["data"]["FPGAtime"][()] if data_pmt and "data" in data_pmt and "FPGAtime" in data_pmt["data"] else np.array([])
    
    timestamps_ch1_all_np = np.array(timestamps_ch1_all_pmt_file)
    coincident_timestamps = np.intersect1d(timestamps_ch1_all_np, timestamps_ch2_all_pmt_file_raw)
    print(f"Number of coincident timestamps (Ch1 & Ch2, all events): {len(coincident_timestamps)}")

    # 4. Force Integration of Ch2 Waveform for a few relevant events
    print(f"\n--- Forced Integration of Ch2 Waveform for selected events ---")
    # Find events with high Ch1 charge to check Ch2 response
    high_ch1_events = df_pmt_events_filtered[df_pmt_events_filtered['ch1_pC'] > 50] # Example threshold, can adjust
    
    if not high_ch1_events.empty:
        sample_integration_timestamps = high_ch1_events['timestamp'].sample(min(3, len(high_ch1_events)), random_state=42).tolist()
        
        for timestamp in sample_integration_timestamps:
            data_file_for_conversion = read_hdffile(PMT_02_FILE_PATH)
            metadata_for_conversion = data_file_for_conversion.get("metadata", {})
            conversion_ch2_val = metadata_for_conversion.get("conversion_ch2", np.array([1.0]))
            if conversion_ch2_val.ndim == 0:
                conversion_ch2_val = conversion_ch2_val.item()

            conversion_factor_pC_val = (1e-6 * (1/60e6) * 1e12)

            _, _, wf2_data_raw, _, ped2_val = get_waveform_at_this_timestamp(PMT_02_FILE_PATH, timestamp)
            
            if wf2_data_raw.size > 0:
                integrated_charge_adc = np.sum(ped2_val - wf2_data_raw)
                
                integrated_charge_pC = integrated_charge_adc * (conversion_ch2_val * conversion_factor_pC_val)
                
                print(f"  Event Timestamp {timestamp} ns (Ch1: {df_pmt_events_filtered[df_pmt_events_filtered['timestamp'] == timestamp]['ch1_pC'].iloc[0]:.2f} pC):")
                print(f"    Ch2 Integrated Charge (ADC units): {integrated_charge_adc:.2f}")
                print(f"    Ch2 Integrated Charge (pC): {integrated_charge_pC:.4f} pC")
            else:
                print(f"  Event Timestamp {timestamp} ns: No waveform data for integration.")
    else:
        print("  No high Ch1 events to sample for Ch2 waveform integration (try lowering filter threshold for this check).")


    # --- Plot 2D Charge Histogram (Ch1 vs Ch2 in PE) ---
    print(f"\n--- Plotting 2D Charge Histogram for {pmt_full_name_for_plot} ---")
    if not df_pmt_events_filtered.empty:
        plot_2d_charge_histogram(
            df_pmt_events_filtered['ch1_pC'], df_pmt_events_filtered['ch2_pC'], pmt_full_name_for_plot, ASSUMED_GAIN, output_file_prefix,
            num_bins=100,
            charge_ch1_ref_pC=df_ref_pmt_events_filtered['ch1_pC'] if not df_ref_pmt_events_filtered.empty else None,
            charge_ch2_ref_pC=df_ref_pmt_events_filtered['ch2_pC'] if not df_ref_pmt_events_filtered.empty else None,
            pmt_full_name_ref=pmt_full_name_ref if not df_ref_pmt_events_filtered.empty else None
        )
    else:
        print("No filtered events available to plot 2D Charge Histogram.")


    # --- Plot Sample Waveforms (general from the main script) ---
    print(f"\n--- Plotting Sample Waveforms for {pmt_full_name_for_plot} ---")
    sample_size = 5
    if not df_pmt_events_filtered.empty and len(df_pmt_events_filtered) >= sample_size:
        sample_timestamps_for_wf = df_pmt_events_filtered['timestamp'].sample(sample_size, random_state=42).tolist()
    elif not df_pmt_events_filtered.empty:
        sample_timestamps_for_wf = df_pmt_events_filtered['timestamp'].tolist() # Less than sample_size filtered events, take all
    else:
        sample_timestamps_for_wf_all, _ = get_timestamps(PMT_02_FILE_PATH, min_charge_pC=0.0, max_charge_pC=float('inf'))
        sample_timestamps_for_wf = sorted(sample_timestamps_for_wf_all[:sample_size]) # No filtered events at all, take first few raw

    plot_sample_waveforms_specific(
        PMT_02_FILE_PATH,
        sample_timestamps_for_wf,
        pmt_full_name_for_plot,
        output_file_prefix
    )

    # --- NEW: Plot Charge Distribution for all 16 PMTs of a LOM ---
    # Determine the base path for all HDF files within the LOM directory
    # For example, if PMT_02_FILE_PATH is "muon_data/data_muon_run_lom16-06/data-muon-run909.09.hdf"
    # then pmt_base_dir_for_all_pmts should be "muon_data/data_muon_run_lom16-06/"
    pmt_base_dir_for_all_pmts = os.path.dirname(PMT_02_FILE_PATH)

    plot_all_lom_pmt_charge_distributions(
        LOM_ID_TO_CHECK,
        pmt_base_dir_for_all_pmts,
        output_base_dir,
        ASSUMED_GAIN,
        ch_name="Ch2"
    )
    # --- END NEW PLOT ---

    print("\n========================================================")
    print("=== Detailed PMT Check Complete. ===")
    print(f"=== Please review the output files in the '{OUTPUT_DIR_NAME}' folder. ===")
    print("========================================================")