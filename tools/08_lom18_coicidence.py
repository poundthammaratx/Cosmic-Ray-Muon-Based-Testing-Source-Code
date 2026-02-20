# 08_lom18_coicidence_nobu.py
#
# Streamlined Comprehensive Muon Analysis Script:
# - Reads HDF files (Muon or SPE data).
# - Generates PMT vs Coincidence Rate plot.
# - Generates Coincidence Matrix (heatmap) for pairwise coincidences.
# - Performs Fast Fourier Transform (FFT) analysis on waveforms of 100 pC peak events.
# - Generates comparison plots (Time-Domain and Frequency-Domain) across all 16 PMTs.
# - Implements LOM18 style ordering, lines, and labels for Coincidence Matrix axes.
# - Calculates livetime based on FPGA timestamps.
# - Allows specifying an output directory for all saved plots.
# - NEW: Waveform preprocessing (Pedestal subtraction, Ch2 polarity inversion) applied.
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

# External libraries (commented out if not explicitly used to avoid unnecessary imports)
# from lmfit import Model
# from matplotlib.colors import LogNorm
# from scipy.interpolate import UnivariateSpline

# === START: Path and Import Management (MODIFIED) ===
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the 'util' subdirectory
util_path = os.path.join(script_dir, "util")
# Add the 'util' directory to sys.sys.path so Python can find modules inside it
if util_path not in sys.path:
    sys.path.append(util_path)

# Import the new, dedicated HDF file reader function from hdf_reader.py
# This function is designed to load the entire HDF file into a dictionary.
from hdf_reader import load_hdf_file_as_dict # <--- MODIFIED import

# Define 'read_hdffile' as an alias for our new, correct loading function.
# This ensures all calls to read_hdffile throughout the script use the right function.
read_hdffile = load_hdf_file_as_dict # <--- MODIFIED alias assignment
# === END: Path and Import Management ===


# Define the mapping for PMT physical positions to subplot indices in a 4x5 grid.
# This mapping ensures plots are arranged according to the LOM16 physical layout.
plotting_map = [6,11,7,12,8,13,9,14,1,16,2,17,3,18,4,19,5,20]


# --- Start: Helper Functions ---

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
    """
    ret_timestamps=[]
    ret_eventidx=[]

    # Now read_hdffile correctly loads the whole file into a dictionary
    data = read_hdffile(filename) 

    # Add checks for missing essential keys in data_dict
    if "data" not in data or "metadata" not in data:
        warnings.warn(f"Missing 'data' or 'metadata' group in {filename}. Skipping timestamp retrieval.")
        return [], []

    # Robustly get data with default empty arrays if keys are missing
    q_ch1 = data["data"]["charge_ch1"][()] if "charge_ch1" in data["data"] else np.array([])
    conversion_ch1 = data["metadata"]["conversion_ch1"][()] if "conversion_ch1" in data["metadata"] else 1.0 # Default to 1.0 if not found
    q_ch1_pC = np.array([ s * (conversion_ch1 * 1e-6 * (1/60e6) * 1e12) for s in q_ch1])

    fpga_time = data["data"]["FPGAtime"][()] if "FPGAtime" in data["data"] else np.array([])
    fpga_time = np.array(fpga_time)

    q_ch2 = data["data"]["charge_ch2"][()] if "charge_ch2" in data["data"] else np.array([])
    conversion_ch2 = data["metadata"]["conversion_ch2"][()] if "conversion_ch2" in data["metadata"] else 1.0 # Default to 1.0 if not found
    q_ch2_pC = np.array([ s * (conversion_ch2 * 1e-6 * (1/60e6) * 1e12) for s in q_ch2])

    # Ensure all arrays have the same length before zipping to prevent IndexError
    min_len = min(len(q_ch1_pC), len(q_ch2_pC), len(fpga_time))

    for iev, (q1, q2, t) in enumerate(zip(q_ch1_pC[:min_len], q_ch2_pC[:min_len], fpga_time[:min_len])):
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

    # Ensure inputs are numpy arrays for efficient indexing
    timestamp_ref_ns = np.asarray(timestamp_ref_ns)
    timestamp_ns = np.asarray(timestamp_ns)

    ptr_ref, ptr_target = 0, 0
    # Iterate through both lists simultaneously using two pointers
    while ptr_ref < len(timestamp_ref_ns) and ptr_target < len(timestamp_ns):
        t_ref = timestamp_ref_ns[ptr_ref]
        t_target = timestamp_ns[ptr_target]

        if abs(t_ref - t_target) <= window_ns:
            ret_times.append([t_ref, t_target])
            ngood += 1
            ptr_ref += 1
            ptr_target += 1
        # If reference timestamp is greater, advance target pointer
        elif t_ref > t_target:
            ptr_target += 1
        # If target timestamp is greater, advance reference pointer
        else: # t_target > t_ref
            ptr_ref += 1

    return ret_times, ngood


def get_waveform_at_this_timestamp(filename, timestamp):
    """
    Retrieves the raw waveform (ADC counts) for a specific timestamp
    from a given HDF file, applying pedestal subtraction and Ch2 inversion.

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

            wf_ch1 = np.array(adc_ch1[iev][:n]) - ped1_val # Pedestal subtraction
            wf_ch2 = np.array(adc_ch2[iev][:n]) - ped2_val # Pedestal subtraction
            
            # --- MODIFIED: Shift Ch1 waveform up by 300 to align with Y-min for plotting ---
            # This is a display-specific shift, not a physical data transformation.
            wf_ch1 = wf_ch1 + 300 

            # Apply Ch2 polarity inversion
            wf_ch2 = wf_ch2 * -1 
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

    if "data" not in data or "metadata" not in data:
        warnings.warn(f"Missing 'data' or 'metadata' group in {filename}. Skipping charge retrieval.")
        return [], []

    # Robustly get data with default empty arrays
    q_ch1 = data["data"]["charge_ch1"][()] if "charge_ch1" in data["data"] else np.array([])
    conversion_ch1 = data["metadata"]["conversion_ch1"][()] if "conversion_ch1" in data["metadata"] else 1.0
    q_ch1_pC_all = np.array([ s * (conversion_ch1 * 1e-6 * (1/60e6) * 1e12) for s in q_ch1]) 

    q_ch2 = data["data"]["charge_ch2"][()] if "charge_ch2" in data["data"] else np.array([])
    conversion_ch2 = data["metadata"]["conversion_ch2"][()] if "conversion_ch2" in data["metadata"] else 1.0
    q_ch2_pC_all = np.array([ s * (conversion_ch2 * 1e-6 * (1/60e6) * 1e12) for s in q_ch2]) 

    for iev in evidx_list: # Iterate through provided event indices directly
        if iev < len(q_ch1_pC_all) and iev < len(q_ch2_pC_all): # Ensure index is valid for existing data
            ret_charges_ch1.append(q_ch1_pC_all[iev])
            ret_charges_ch2.append(q_ch2_pC_all[iev])
        else:
            ret_charges_ch1.append(np.nan) # Append NaN if index is out of bounds
            ret_charges_ch2.append(np.nan)
            warnings.warn(f"Event index {iev} out of bounds for charge data in {filename}. Appending NaN.")

    return ret_charges_ch1, ret_charges_ch2
# --- End: Helper Functions ---


# --- Start: Plotting Functions ---

def plot_multi_pmt_waveforms_comparison(pmt_wf_data_collection, plot_title, output_filename, output_dir):
    """
    Plots waveforms for multiple PMTs in a single figure with 4x4 subplots.
    pmt_wf_data_collection: Dict like {pmt_id: {'x_wf': array, 'wf_data': array, 'ch_label': str, 'charge_pC': float}}
    """
    fig, axes = plt.subplots(4, 4, figsize=(15, 12)) # 4x4 grid for 16 subplots
    axes = axes.flatten() # Flatten the 2D array of axes for easy iteration

    # Define the custom order of PMTs for the 4x4 subplot grid as requested
    custom_subplot_order_for_4x4_grid = np.array([
        8, 10, 12, 14,
        2,  6,  8, 10, # Note: PMT 8 and 10 are repeated here in your original mapping
        1,  3,  5,  7,
        9, 11, 13, 15
    ])

    # Determine global Y-limit for ADC counts for consistent comparison across subplots
    global_ymin_adc = 0 # Adjusted lower limit to accommodate negative ADC values after pedestal subtraction
    global_ymax_adc = 4095 # Assuming a 12-bit ADC, typical range is 0-4095

    # Determine global X-limit for time (ns) if necessary, usually fixed by DAQ window
    global_xmin_ns = float('inf')
    global_ymax_ns = float('-inf')
    # Loop through the *unique* PMT IDs that are actually present in the data collection
    # to find the overall time limits for consistent plotting.
    for pmt_id in pmt_wf_data_collection.keys():
        data = pmt_wf_data_collection.get(pmt_id)
        if data and data['x_wf'] is not None and len(data['x_wf']) > 1: # Check if data exists and is valid
            global_xmin_ns = min(global_xmin_ns, data['x_wf'][0])
            global_ymax_ns = max(global_ymax_ns, data['x_wf'][-1])

    # If no valid data found across all PMTs, set default reasonable range for consistency
    if global_xmin_ns == float('inf'): global_xmin_ns = 0
    if global_ymax_ns == float('-inf'): global_ymax_ns = 1000 # Example default 1000 ns (for 60 MSPS, ~17us max)

    # Plot each PMT's waveform in its respective subplot based on the custom order
    # Loop through the custom order, and use the PMT ID at each position to fetch data.
    for i, pmt_id_to_plot in enumerate(custom_subplot_order_for_4x4_grid):
        ax = axes[i] # Get the current subplot axis
        data = pmt_wf_data_collection.get(pmt_id_to_plot) # Fetch data for the specific PMT ID

        # Check if data exists for this PMT ID (it might be None if no events in peak for that PMT)
        if data and data['wf_data'] is not None and len(data['wf_data']) > 0: 
            ax.plot(data['x_wf'], data['wf_data'], color='blue')
            ax.set_title(f"PMT {pmt_id_to_plot} {data['ch_label']} ({data['charge_pC']:.1f}pC)", fontsize=8)
            ax.set_xlabel("Time (ns)", fontsize=7)
            ax.set_ylabel("ADC Counts", fontsize=7)
            ax.set_xlim([global_xmin_ns, global_ymax_ns]) # Apply consistent x-limits
            ax.set_ylim([global_ymin_adc, global_ymax_adc]) 
        else:
            # If no data for this PMT ID, display a placeholder text
            ax.text(0.5, 0.5, f"PMT {pmt_id_to_plot}\nNo Data in Peak", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_title(f"PMT {pmt_id_to_plot}", fontsize=8)
            ax.set_xticks([]) # Hide ticks if no plot
            ax.set_yticks([])

        ax.grid(True, linestyle=':', lw=0.5) # Add grid for readability
        ax.tick_params(labelsize=6) # Adjust tick label font size

    # Hide any unused subplots (though for a 4x4 grid of 16 elements, all should be used)
    for j in range(len(custom_subplot_order_for_4x4_grid), 16):
        fig.delaxes(axes[j]) # Remove empty subplot

    fig.suptitle(plot_title, fontsize=14) # Set overall title for the figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title/label overlap
    plt.savefig(os.path.join(output_dir, output_filename)) # Save the figure
    plt.close(fig) # Close the figure to free up memory
    print(f'Saved {os.path.join(output_dir, output_filename)}')


# Function to plot a multi-subplot comparison for FFTs
def plot_multi_pmt_ffts_comparison(pmt_fft_data_collection, plot_title, output_filename, output_dir):
    """
    Plots FFT magnitude spectra for multiple PMTs in a single figure with 4x4 subplots.
    pmt_fft_data_collection: Dict like {pmt_id: {'frequencies': array, 'magnitude': array, 'ch_label': str, 'charge_pC': float}}
    """
    fig, axes = plt.subplots(4, 4, figsize=(15, 12)) # 4x4 grid for 16 subplots
    axes = axes.flatten() # Flatten the 2D array of axes

    # Define the custom order of PMTs for the 4x4 subplot grid as requested
    custom_subplot_order_for_4x4_grid = np.array([
        8, 10, 12, 14,
        2,  6,  8, 10, # Note: PMT 8 and 10 are repeated here in your original mapping
        1,  3,  5,  7,
        9, 11, 13, 15
    ])

    # Determine global X-limit for frequency (MHz) based on actual data ranges
    global_xmax_freq_mhz = float('-inf')
    # Loop through the *unique* PMT IDs that are actually present in the data collection
    for pmt_id in pmt_fft_data_collection.keys():
        data = pmt_fft_data_collection.get(pmt_id)
        if data and data['frequencies'] is not None and len(data['frequencies']) > 1: # Check if data exists and is valid
            global_xmax_freq_mhz = max(global_xmax_freq_mhz, (data['frequencies'] / 1e6)[-1])
    # If no valid data, set a default Nyquist frequency for 60 MSPS (Sampling Rate / 2)
    if global_xmax_freq_mhz == float('-inf'): global_xmax_freq_mhz = 30 

    # Plot each PMT's FFT spectrum in its respective subplot based on the custom order
    for i, pmt_id_to_plot in enumerate(custom_subplot_order_for_4x4_grid):
        ax = axes[i]
        data = pmt_fft_data_collection.get(pmt_id_to_plot) # Fetch data for the specific PMT ID

        if data and data['magnitude'] is not None and len(data['magnitude']) > 0: # Check for valid magnitude data
            ax.plot(data['frequencies'] / 1e6, data['magnitude'], color='red') # Plot frequency in MHz
            ax.set_title(f"PMT {pmt_id_to_plot} {data['ch_label']} ({data['charge_pC']:.1f}pC)", fontsize=8)
            ax.set_xlabel("Frequency (MHz)", fontsize=7)
            ax.set_ylabel("Magnitude", fontsize=7)
            ax.set_xlim([0, global_xmax_freq_mhz]) # Apply consistent x-limits
            ax.set_yscale('log') # Log scale for magnitude is often very useful for FFT spectra
        else:
            # If no data for this PMT ID, display a placeholder text
            ax.text(0.5, 0.5, f"PMT {pmt_id_to_plot}\nNo Data in Peak", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_title(f"PMT {pmt_id_to_plot}", fontsize=8)
            ax.set_xticks([]) # Hide ticks if no plot
            ax.set_yticks([])

        ax.grid(True, linestyle=':', lw=0.5) # Add grid
        ax.tick_params(labelsize=6) # Adjust tick label font size

    # Hide any unused subplots
    for j in range(len(custom_subplot_order_for_4x4_grid), 16):
        fig.delaxes(axes[j])

    fig.suptitle(plot_title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, output_filename))
    plt.close(fig)
    print(f'Saved {os.path.join(output_dir, output_filename)}')


if __name__ == "__main__":
    # Configure command-line argument parser
    parser = argparse.ArgumentParser(description="Streamlined Muon Analysis Script: Focuses on coincidence and FFT analysis.")
    parser.add_argument("infiles", nargs='+', help="Input HDF file names (e.g., 'muon_data/data_muon_run_*.hdf')", type=str)
    parser.add_argument("--output_dir", "-o", help="Output directory for plots and tabular data (default: ./plots_output)", type=str, default='./plots_output') # New default output directory
    # Added argument for assumed gain, which is used for 2 PE threshold calculation
    parser.add_argument("--assumed_gain", type=float, default=5e6, help="Assumed PMT gain for PE conversion (electrons/PE). Default is 5e6.") # New argument
    args = parser.parse_args()

    # --- Start: File Scanning and PMT Map Creation ---
    infilenames = [] # List to store paths of all found HDF files
    # Expand wildcards in input file paths (e.g., 'muon_data/data_muon_run_*.hdf')
    for f in args.infiles:
        expanded_files = glob.glob(f) 
        for infile_path in expanded_files:
            # Check if it's a file and has .hdf extension
            if os.path.isfile(infile_path) and infile_path.lower().endswith(".hdf"):
                infilenames.append(infile_path)

    if not infilenames: 
        print("Error: No HDF files found matching the pattern. Please check input path and filename extensions.")
        sys.exit(1) # Exit if no files are found

    print(f"Found {len(infilenames)} files: {infilenames}")

    runid=None # Variable to store run ID
    lomname=None # Variable to store LOM name
    pmt_to_infile_map = {} # Dictionary to map PMT ID to its corresponding HDF file path

    for fname in infilenames:
        # Split path to get directory and file name components
        path_parts = os.path.split(fname) 

        # Robust parsing for LOM name (Location of Module)
        # Attempt to extract LOM name from parent directory.
        if len(path_parts) >= 2:
            this_lom_candidate = os.path.basename(path_parts[-2]) 
            if 'data' in this_lom_candidate.lower() or 'run' in this_lom_candidate.lower():
                parent_dir_of_candidate = os.path.basename(os.path.dirname(fname)) 
                if parent_dir_of_candidate and 'data' not in parent_dir_of_candidate.lower() and 'run' not in parent_dir_of_candidate.lower():
                    this_lom_candidate = parent_dir_of_candidate # Use the higher-level directory as LOM name

            # Final check to ensure the candidate name is not a generic 'data' or 'run' folder
            if 'data' not in this_lom_candidate.lower() and 'run' not in this_lom_candidate.lower(): 
                if lomname is None:
                    lomname = this_lom_candidate # Assign if it's the first LOM name found
                elif lomname != this_lom_candidate:
                    warnings.warn(f'LOM name inconsistency found: {this_lom_candidate} {lomname}' ) # Warn if LOM names are inconsistent

        # Parsing for Run ID from filename (e.g., 'data_muon_run909_00.hdf' -> 909)
        file_base = os.path.basename(fname) 
        run_parts = file_base.split('_')
        if len(run_parts) >= 3:
            try:
                # Extract run ID, removing 'run' prefix if present
                current_runid = int(run_parts[-2].replace('run',''))
                if runid is None:
                    runid = current_runid
                elif runid != current_runid:
                    warnings.warn(f'Run ID inconsistency found: {current_runid} {runid}' )
            except (ValueError, IndexError):
                pass # Ignore if run ID cannot be parsed

        # Populate pmt_to_infile_map: Map PMT ID to its HDF file path
        try:
            ch_str_parts = file_base.split(".")
            pmt_id = -1
            # Attempt to parse PMT ID from different parts of the filename
            if len(ch_str_parts) >= 2 and ch_str_parts[-2].isdigit():
                pmt_id = int(ch_str_parts[-2])
            elif len(file_base.split("_")) >= 2 and file_base.split("_")[-1].split(".")[0].isdigit():
                pmt_id = int(file_base.split("_")[-1].split(".")[0])

            # Ensure parsed PMT ID is within the expected range for plotting_map (0-19)
            # The original plotting_map has 18 elements, PMT IDs 0-15 are common.
            # Adjust condition to ensure pmt_id is a valid index or within expected range
            if 0 <= pmt_id <= 15: # Assuming PMT IDs are 0-15 for the 16 PMTs
                pmt_to_infile_map[pmt_id] = fname
            else:
                warnings.warn(f"Parsed PMT ID {pmt_id} from {fname} is out of expected range (0-15). Skipping for PMT map.")

        except (ValueError, IndexError):
            warnings.warn(f"Could not parse PMT ID from filename: {fname}. Skipping for PMT map.")

    # Print parsed LOM name and Run ID if found
    if lomname:
        print(f'This LOM is: {lomname}')
    if runid is not None:
        print(f'Run ID: {runid}')
    # --- End: File Scanning and PMT Map Creation ---


    # --- Start: Coincidence Analysis ---
    print("\n--- Starting Coincidence Analysis ---")
    # Define parameters for coincidence detection (can be made configurable via argparse)
    # Calculate 2 PE threshold based on assumed gain
    electron_charge_C = 1.602e-19
    charge_per_pe_C = args.assumed_gain * electron_charge_C
    charge_per_pe_pC = charge_per_pe_C * 1e12 # pC per PE

    min_charge_pC_for_coincidence = 2.0 * charge_per_pe_pC # Set to 2 PE threshold based on assumed gain
    max_charge_pC_for_coincidence = float('inf') # No upper limit specified for coincidence threshold in image. Set to inf or a very high value.
    coincidence_window_ns = 100.0 # Time window in nanoseconds for events to be coincident (from image title)

    all_pmt_event_times = {} # Dictionary to store filtered timestamps for each PMT
    all_pmt_livetimes = {} # Dictionary to store calculated livetime for each PMT

    print("Calculating Livetimes and Filtering Events for Coincidence...")
    pmt_ids_with_data = sorted(pmt_to_infile_map.keys()) # Get sorted list of PMT IDs for which files were found

    # Populate event times and calculate livetimes for all PMTs
    for pmt_id in pmt_ids_with_data:
        infile_pmt = pmt_to_infile_map[pmt_id]

        data_all_fpga = read_hdffile(infile_pmt) # This now uses load_hdf_file_as_dict

        # Robustly handle missing "FPGAtime" key by providing an empty array as fallback
        all_raw_fpga_times = np.array(data_all_fpga["data"]["FPGAtime"][()] if "data" in data_all_fpga and "FPGAtime" in data_all_fpga["data"] else [])

        pmt_livetime = 0.0
        if all_raw_fpga_times.size > 1:
            start_t = all_raw_fpga_times[0]
            end_t = all_raw_fpga_times[-1]
            pmt_livetime = (end_t - start_t) / 1e9 # Convert from ns to seconds
            if pmt_livetime <= 0: pmt_livetime = 0.0 # Handle cases with non-positive livetime

        all_pmt_livetimes[pmt_id] = pmt_livetime

        # Get timestamps of events passing the charge threshold range for coincidence (for either ch1 or ch2)
        filtered_timestamps, _ = get_timestamps(infile_pmt, min_charge_pC=min_charge_pC_for_coincidence, max_charge_pC=max_charge_pC_for_coincidence)
        all_pmt_event_times[pmt_id] = np.sort(np.array(filtered_timestamps)) # Ensure timestamps are sorted

        print(f"PMT {pmt_id}: Filtered Events = {len(filtered_timestamps)}, Calculated Livetime = {pmt_livetime:.2f}s")


    print("\nBuilding Coincidence Matrix (Heatmap)...")
    # Determine the maximum PMT ID found in the data to size the matrices correctly
    max_pmt_id_in_data = max(pmt_ids_with_data) if pmt_ids_with_data else -1

    if max_pmt_id_in_data == -1:
        print("No valid PMT data found to build coincidence matrix. Skipping.")
    else:
        coincidence_matrix_counts = np.zeros((max_pmt_id_in_data + 1, max_pmt_id_in_data + 1), dtype=int)
        coincidence_matrix_rates = np.zeros((max_pmt_id_in_data + 1, max_pmt_id_in_data + 1), dtype=float)

        # Calculate pairwise coincidences between all PMTs
        for i_pmt in pmt_ids_with_data:
            for j_pmt in pmt_ids_with_data:
                if i_pmt == j_pmt: # Skip self-coincidence for the pairwise matrix
                    continue

                times_i = all_pmt_event_times.get(i_pmt, np.array([]))
                times_j = all_pmt_event_times.get(j_pmt, np.array([]))

                # Only attempt matching if both PMTs have events
                if times_i.size > 0 and times_j.size > 0:
                    _, count = event_matching(times_i, times_j, window_ns=coincidence_window_ns)
                else:
                    count = 0

                coincidence_matrix_counts[i_pmt, j_pmt] = count

                livetime_i = all_pmt_livetimes.get(i_pmt, 0.0)
                livetime_j = all_pmt_livetimes.get(j_pmt, 0.0)

                # Use the minimum of the two livetimes for pairwise rate calculation (common practice)
                common_livetime = min(livetime_i, livetime_j) 
                if common_livetime > 0:
                    coincidence_matrix_rates[i_pmt, j_pmt] = count / common_livetime
                else:
                    coincidence_matrix_rates[i_pmt, j_pmt] = 0.0 # Rate is 0 if no common livetime

        print("Coincidence Counts Matrix:")
        print(coincidence_matrix_counts)
        print("\nCoincidence Rates Matrix (Hz):")
        print(coincidence_matrix_rates)

        # --- LOM18 Style Ordering for Coincidence Matrix ---
        # Define the specific physical order of PMT IDs for the LOM18 style axes.
        #lom18_x_labels_order = np.array([14,12,10,8,6,4,2,0,1,3,5,7,9,11,13,15])
        #lom18_y_labels_order = np.array([15,13,11,9,7,5,3,1,0,2,4,6,8,10,12,14]) # Consistent with your request for bottom-up labels
        
        lom18_y_labels_order = np.array([14,12,10,8,6,4,2,0,1,3,5,7,9,11,13,15])
        lom18_x_labels_order = np.array([15,13,11,9,7,5,3,1,0,2,4,6,8,10,12,14])


        # Filter these lists to only include PMT IDs that are actually present in the data (0-15)
        # This prevents indexing errors if some PMTs are missing
        lom18_x_indices_filtered = [p for p in lom18_x_labels_order if p in pmt_ids_with_data]
        lom18_y_indices_filtered = [p for p in lom18_y_labels_order if p in pmt_ids_with_data]

        if not lom18_x_indices_filtered or not lom18_y_indices_filtered:
            print("Not enough PMT data to create LOM18 style coincidence matrix plot.")
        else:
            # Reorder the coincidence matrices based on the LOM18 physical order.
            coincidence_matrix_rates_ordered = coincidence_matrix_rates[np.ix_(lom18_y_indices_filtered, lom18_x_indices_filtered)]
            coincidence_matrix_counts_ordered = coincidence_matrix_counts[np.ix_(lom18_y_indices_filtered, lom18_x_indices_filtered)]


            fig_matrix = plt.figure(figsize=(10, 8)) # Create figure for coincidence matrix
            ax_matrix = fig_matrix.add_subplot(111)

            # Plot the reordered coincidence matrix using pcolormesh.
            mesh = ax_matrix.pcolormesh(np.arange(len(lom18_x_indices_filtered)), 
                                        np.arange(len(lom18_y_indices_filtered)), 
                                        coincidence_matrix_rates_ordered, 
                                        cmap='Blues', edgecolors='white', linewidth=1.0) 
            fig_matrix.colorbar(mesh, ax=ax_matrix, label='Coincidence Rate [Hz]') # Add colorbar

            # Set ticks and labels according to the ordered PMT IDs for both X and Y axes.
            ax_matrix.set_xticks(np.arange(len(lom18_x_indices_filtered)) + 0.5) 
            ax_matrix.set_yticks(np.arange(len(lom18_y_indices_filtered)) + 0.5)
            ax_matrix.set_xticklabels(lom18_x_indices_filtered)
            ax_matrix.set_yticklabels(lom18_y_indices_filtered) 

            ax_matrix.set_xlabel('Channel ID', fontsize=12)
            ax_matrix.set_ylabel('Channel ID', fontsize=12)

            # Main title is now combined to avoid overlap as per the image example
            fig_matrix.suptitle(f"2 PE threshold\nCoincidence rate (ΔT = {coincidence_window_ns} ns) [Hz]", fontsize=14, y=1.02) # y=1.02 pushes it slightly above

            # Invert Y-axis for typical matrix display (origin at top-left) which matches the reference image
            ax_matrix.invert_yaxis() 
            ax_matrix.set_aspect('equal', adjustable='box') # Ensure cells are square

            # --- Add Hemisphere Dividing Lines and Labels ---
            try:
                # Find the index of PMT 0 in the *filtered* display orders.
                split_idx_x = lom18_x_indices_filtered.index(0)
                split_idx_y = lom18_y_indices_filtered.index(0) 

                ax_matrix.axvline(split_idx_x + 0.5, color='black', linestyle='-', linewidth=2) 
                ax_matrix.axhline(split_idx_y + 0.5, color='black', linestyle='-', linewidth=2)

                # Add "← upper" and "lower →" labels as text annotations
                ax_matrix.text(0.5 * (split_idx_x + 0.5) / len(lom18_x_indices_filtered), -0.08, '← upper', 
                                color='black', fontsize=10, ha='center', va='top', transform=ax_matrix.transAxes)
                ax_matrix.text(( (split_idx_x + 0.5) + (len(lom18_x_indices_filtered) - (split_idx_x + 0.5)) ) / 2 / len(lom18_x_indices_filtered), -0.08, 'lower →', 
                                color='black', fontsize=10, ha='center', va='top', transform=ax_matrix.transAxes)

                ax_matrix.text(-0.08, ( (split_idx_y + 0.5) + (len(lom18_y_indices_filtered) - (split_idx_y + 0.5)) ) / 2 / len(lom18_y_indices_filtered), '← upper', 
                                color='black', fontsize=10, ha='right', va='center', rotation=90, transform=ax_matrix.transAxes)
                ax_matrix.text(-0.08, (split_idx_y + 0.5) / 2 / len(lom18_y_indices_filtered), 'lower →', 
                                color='black', fontsize=10, ha='right', va='center', rotation=90, transform=ax_matrix.transAxes)

            except ValueError:
                warnings.warn("PMT 0 not found in filtered axis order. Cannot draw hemisphere lines/labels precisely.")

            if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
            # Save the LOM18-styled coincidence matrix plot with a distinct filename
            fig_matrix.savefig(os.path.join(args.output_dir, "muon_coincidence_matrix_LOM18_style.pdf")) 
            plt.close(fig_matrix)
            print(f'Saved {os.path.join(args.output_dir, "muon_coincidence_matrix_LOM18_style.pdf")}')


            print("\nPlotting PMT vs Total Coincidence Rate...")
            # Calculate total coincidence rates per PMT (summing across rows of the original matrix)
            total_coincidence_rates_per_pmt = np.sum(coincidence_matrix_rates, axis=1) 

            fig_rate_total = plt.figure(figsize=(10, 6))
            ax_rate_total = fig_rate_total.add_subplot(111)
            # Filter PMTs to plot based on actual data presence and max ID
            pmts_to_plot_rate = [p for p in pmt_ids_with_data if p <= max_pmt_id_in_data] 
            # Scatter plot of PMT ID vs. Total Coincidence Rate
            ax_rate_total.scatter(pmts_to_plot_rate, total_coincidence_rates_per_pmt[pmts_to_plot_rate], color='blue', s=50)
            ax_rate_total.set_xlabel('PMT Channel ID',fontsize=10)
            ax_rate_total.set_ylabel('Total Coincidence Rate [1/s]',fontsize=10)
            ax_rate_total.set_title(f'Total Coincidence Rate per PMT (2 PE threshold, Window={coincidence_window_ns}ns)', fontsize=12) # Use hardcoded label as requested
            ax_rate_total.tick_params(labelsize=10)
            ax_rate_total.grid('xy', linestyle=':', lw=0.5)
            plt.tight_layout()
            # Save the total coincidence rate plot
            fig_rate_total.savefig(os.path.join(args.output_dir, "muon_total_coincidence_rate_plot.pdf")) 
            plt.close(fig_rate_total)
            print(f'Saved {os.path.join(args.output_dir, "muon_total_coincidence_rate_plot.pdf")}')

    # --- Start: FFT Analysis of 100 pC Peak Waveforms and Comparison Plots ---
    print("\n--- Starting FFT Analysis of 100 pC Peak Waveforms ---")
    min_charge_peak_pC = 80.0 # Lower bound of charge range for peak analysis
    max_charge_peak_pC = 120.0 # Upper bound of charge range for peak analysis

    # Dictionaries to store one representative event's data per PMT for comparison plots
    pmt_ch1_wf_data_for_comparison = {} # Stores Time-Domain waveform data for Ch1
    pmt_ch1_fft_data_for_comparison = {} # Stores Frequency-Domain FFT data for Ch1
    pmt_ch2_wf_data_for_comparison = {} # Stores Time-Domain waveform data for Ch2
    pmt_ch2_fft_data_for_comparison = {} # Stores Frequency-Domain FFT FFT data for Ch2

    # Loop through each PMT that has data
    for pmt_id in pmt_ids_with_data: 
        infile_pmt = pmt_to_infile_map[pmt_id]

        # Get timestamps and event indices for events whose charges fall within the 100 pC peak range
        timestamps_in_peak, event_indices_in_peak = get_timestamps(
            infile_pmt,
            min_charge_pC=min_charge_peak_pC,
            max_charge_pC=max_charge_peak_pC
        )

        if not timestamps_in_peak:
            print(f"PMT {pmt_id}: No events found in {min_charge_peak_pC}-{max_charge_peak_pC} pC range. Skipping FFT and comparison for this PMT.")
            # Populate comparison dicts with None to indicate no data for this PMT/channel
            pmt_ch1_wf_data_for_comparison[pmt_id] = None
            pmt_ch1_fft_data_for_comparison[pmt_id] = None
            pmt_ch2_wf_data_for_comparison[pmt_id] = None
            pmt_ch2_fft_data_for_comparison[pmt_id] = None
            continue

        # Select the first event found in the peak range as a representative for this PMT
        representative_event_idx = event_indices_in_peak[0]
        representative_timestamp = timestamps_in_peak[0]

        # Get the actual charge values for the representative event
        rep_charges_ch1, rep_charges_ch2 = get_charges_of_these_events(infile_pmt, [representative_event_idx])
        rep_charge_ch1_pC = rep_charges_ch1[0]
        rep_charge_ch2_pC = rep_charges_ch2[0]

        print(f"PMT {pmt_id}: Analyzing representative Event {representative_event_idx} (Time: {representative_timestamp} ns, Ch1 Charge: {rep_charge_ch1_pC:.2f} pC, Ch2 Charge: {rep_charge_ch2_pC:.2f} pC)")

        # Get the waveform data for the representative event
        x_wf, wf1_data, wf2_data = get_waveform_at_this_timestamp(infile_pmt, representative_timestamp)

        # Calculate sampling interval (dt) and sampling frequency (Fs) from the waveform data
        dt_ns = x_wf[1] - x_wf[0] if len(x_wf) > 1 else (1e9 / 60e6) # Fallback if only one sample
        dt_s = dt_ns * 1e-9 # Convert to seconds for FFT calculations
        Fs_Hz = 1.0 / dt_s # Sampling frequency in Hz

        # Process Channel 1 data
        if len(wf1_data) > 0:
            N_ch1 = len(wf1_data)
            fft_result_ch1 = scipy.fft.fft(wf1_data)
            frequencies_ch1 = scipy.fft.fftfreq(N_ch1, d=dt_s)
            magnitude_spectrum_ch1 = np.abs(fft_result_ch1[:N_ch1 // 2]) # Take only positive frequencies
            pmt_ch1_wf_data_for_comparison[pmt_id] = {
                'x_wf': x_wf, 'wf_data': wf1_data, 'ch_label': "Ch1", 'charge_pC': rep_charge_ch1_pC
            }
            pmt_ch1_fft_data_for_comparison[pmt_id] = {
                'frequencies': frequencies_ch1[:N_ch1 // 2], 'magnitude': magnitude_spectrum_ch1, 
                'ch_label': "Ch1", 'charge_pC': rep_charge_ch1_pC
            }
        else:
            print(f"   PMT {pmt_id} Ch1: No waveform data for representative event. Skipping comparison data.")
            pmt_ch1_wf_data_for_comparison[pmt_id] = None
            pmt_ch1_fft_data_for_comparison[pmt_id] = None

        # Process Channel 2 data
        if len(wf2_data) > 0:
            N_ch2 = len(wf2_data)
            fft_result_ch2 = scipy.fft.fft(wf2_data)
            frequencies_ch2 = scipy.fft.fftfreq(N_ch2, d=dt_s)
            magnitude_spectrum_ch2 = np.abs(fft_result_ch2[:N_ch2 // 2]) # Take only positive frequencies
            pmt_ch2_wf_data_for_comparison[pmt_id] = {
                'x_wf': x_wf, 'wf_data': wf2_data, 'ch_label': "Ch2", 'charge_pC': rep_charge_ch2_pC
            }
            pmt_ch2_fft_data_for_comparison[pmt_id] = {
                'frequencies': frequencies_ch2[:N_ch2 // 2], 'magnitude': magnitude_spectrum_ch2, 
                'ch_label': "Ch2", 'charge_pC': rep_charge_ch2_pC
            }
        else:
            print(f"   PMT {pmt_id} Ch2: No waveform data for representative event. Skipping comparison data.")
            pmt_ch2_wf_data_for_comparison[pmt_id] = None
            pmt_ch2_fft_data_for_comparison[pmt_id] = None


    print("\n--- Generating Comparison Plots for all PMTs ---")
    output_dir_comparison = args.output_dir # Use main output directory for these comparison plots

    # Plot Comparison Waveforms (Ch1)
    plot_multi_pmt_waveforms_comparison(
        pmt_ch1_wf_data_for_comparison,
        "Time Domain Waveforms: Ch1 (Representative 100 pC Peak Events)",
        "all_pmts_time_domain_ch1_comparison.pdf",
        output_dir_comparison
    )

    # Plot Comparison Waveforms (Ch2)
    plot_multi_pmt_waveforms_comparison(
        pmt_ch2_wf_data_for_comparison,
        "Time Domain Waveforms: Ch2 (Representative 100 pC Peak Events)",
        "all_pmts_time_domain_ch2_comparison.pdf",
        output_dir_comparison
    )

    # Plot Comparison FFTs (Ch1)
    plot_multi_pmt_ffts_comparison(
        pmt_ch1_fft_data_for_comparison,
        "Frequency Domain Spectra: Ch1 (Representative 100 pC Peak Events)",
        "all_pmts_freq_domain_ch1_comparison.pdf",
        output_dir_comparison
    )

    # Plot Comparison FFTs (Ch2)
    plot_multi_pmt_ffts_comparison(
        pmt_ch2_fft_data_for_comparison,
        "Frequency Domain Spectra: Ch2 (100 pC Peak Events)",
        "all_pmts_freq_domain_ch2_comparison.pdf",
        output_dir_comparison
    )

    print("\n--- FFT Analysis and Comparison Plots Complete ---")
    # --- End: FFT Analysis of 100 pC Peak Waveforms ---

    plt.show() # Display all generated plots (if running interactively)