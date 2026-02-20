# waveform_fft_analysis.py
import h5py
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit # Still needed for linear_function in get_charges... if used or could be removed
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

# === START: Path and Import Management (Adjust as needed for your 'util' folder location) ===
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the 'util' subdirectory.
# IMPORTANT: If 'hdf_reader.py' is not in a 'util' folder next to this script,
# you will need to adjust 'util_path' or ensure 'hdf_reader.py' is in sys.path.
util_path = os.path.join(script_dir, "util") # Assumes 'util' is a subdirectory
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


# --- Start: Helper Functions (Only those required for waveform and charge retrieval) ---

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


# --- Start: Plotting Functions (Copied directly as they were, but now standalone) ---

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
            ax.set_ylabel("Ch1 ADC Counts", fontsize=12)
            ax.set_xlim([xmin_ns_current, xmax_ns_current])
            ax.set_ylim([ymin_adc, ymax_adc])
            #ax.tick_params(labelsize = 15)

            # Add cutting line for Ch1 at ADC 300
            if "Ch1" in plot_title:
                ax.axhline(y=300, color='orange', linestyle='--', linewidth=0.8, alpha=0.7)
                ax.axhline(y=4095, color='purple', linestyle='-', linewidth=0.8, alpha=0.7)
                # No legend label here, as it's typically handled by a general legend or implied
            # Add cutting lines for Ch2 based on new interpretation (raw ADC, high baseline, negative pulse)
            elif "Ch2" in plot_title:
                ax.axhline(y=3800, color='gray', linestyle='-', linewidth=0.8, alpha=0.7, label='Baseline (~3800)')
                ax.axhline(y=3500, color='green', linestyle='--', linewidth=0.8, alpha=0.7, label='ADC Cut (3500)')
                ax.legend(fontsize=6, loc='upper right') # Only legend in Ch2 plots to avoid clutter
        else:
            ax.text(0.5, 0.5, f"PMT {pmt_id_to_plot}\nNo Data in Peak", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_title(f"PMT {pmt_id_to_plot}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        ax.grid(True, linestyle=':', lw=0.5)
        ax.tick_params(labelsize=12)

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
        if j < len(axes): # Ensure axis exists before deleting
            fig.delaxes(axes[j])

    fig.suptitle(plot_title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, output_filename))
    plt.close(fig)
    print(f'Saved {os.path.join(output_dir, output_filename)}')


# --- End: Plotting Functions ---


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone script for Waveform and FFT analysis of PMT data.")
    parser.add_argument("lom_data_dirs", nargs='+', help="Input LOM data directories (e.g., 'path/to/LOM16-01/'). Each directory should contain 16 PMT HDF files.", type=str)
    parser.add_argument("--output_dir", "-o", help="Output directory for plots (default: ./plots_output_waveforms)", type=str, default='./plots_output_waveforms')
    args = parser.parse_args()

    # Define the peak charge range for 'Bright Events' waveform analysis
    min_charge_bright_signal_pC = 80.0
    max_charge_bright_signal_pC = 120.0

    # Define the charge range for 'Small Signal' waveform analysis
    min_charge_small_signal_pC = 5.0  # Example: adjust as needed
    max_charge_small_signal_pC = 20.0 # Example: adjust as needed

    # --- Create Output Directories ---
    output_base_dir = args.output_dir
    os.makedirs(output_base_dir, exist_ok=True)

    output_dirs = {
        "waveform_analysis_ch1": os.path.join(output_base_dir, "waveform_analysis_ch1"),
        "waveform_analysis_ch2": os.path.join(output_base_dir, "waveform_analysis_ch2"),
        "fft_analysis_ch1": os.path.join(output_base_dir, "fft_analysis_ch1"),
        "fft_analysis_ch2": os.path.join(output_base_dir, "fft_analysis_ch2"),
    }
    for d in output_dirs.values():
        os.makedirs(d, exist_ok=True)
    print(f"Output plots will be saved to: {output_base_dir} and its subfolders.")

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

        if not pmt_to_infile_map or len(pmt_to_infile_map) < 16:
            warnings.warn(f"Warning: Only {len(pmt_to_infile_map)} PMT files found for LOM {current_lom_name}. Waveform/FFT plots may be incomplete or skipped.")

        # Prepare LOM-specific prefix for output filenames
        lom_output_prefix = current_lom_name
        if current_lom_runid is not None:
            lom_output_prefix = f"{current_lom_name}_Run{current_lom_runid}_"
        else:
            lom_output_prefix = f"{current_lom_name}_" # Fallback if runid not found

        # --- Start: FFT Analysis of Waveforms and Comparison Plots ---
        print("\n--- Starting Waveform and FFT Analysis ---")

        pmt_ch1_wf_data_for_comparison = {}
        pmt_ch1_fft_data_for_comparison = {}
        pmt_ch2_wf_data_for_comparison = {}
        pmt_ch2_fft_data_for_comparison = {}

        pmt_ids_with_data_sorted = sorted(pmt_to_infile_map.keys())

        for pmt_id in pmt_ids_with_data_sorted:
            infile_pmt = pmt_to_infile_map[pmt_id]

            current_min_charge = min_charge_bright_signal_pC
            current_max_charge = max_charge_bright_signal_pC
            charge_range_description = f"{min_charge_bright_signal_pC:.1f}-{max_charge_bright_signal_pC:.1f} pC (Bright Signal)"

            # Apply specific charge ranges based on PMT ID
            if pmt_id in [0, 1, 8, 9]:
                # PMT 00, 01, 08, 09: Bright Events (already set by default)
                pass
            else:
                # Other PMTs: Small Signal Events
                current_min_charge = min_charge_small_signal_pC
                current_max_charge = max_charge_small_signal_pC
                charge_range_description = f"{min_charge_small_signal_pC:.1f}-{max_charge_small_signal_pC:.1f} pC (Small Signal)"

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

        # Plot Ch1 Waveforms (Peak Focus)
        plot_multi_pmt_waveforms_comparison(
            pmt_ch1_wf_data_for_comparison,
            f"Time Domain Waveforms: Ch1 (Representative Events) - LOM {current_lom_name}",
            f"{lom_output_prefix}all_pmts_time_domain_ch1_comparison", # Base filename
            output_dirs["waveform_analysis_ch1"],
            x_range_mode="peak_focus"
        )

        # Plot Ch1 Waveforms (Broad View)
        plot_multi_pmt_waveforms_comparison(
            pmt_ch1_wf_data_for_comparison,
            f"Time Domain Waveforms: Ch1 (Representative Events) - LOM {current_lom_name}",
            f"{lom_output_prefix}all_pmts_time_domain_ch1_comparison", # Base filename
            output_dirs["waveform_analysis_ch1"],
            x_range_mode="broad"
        )

        # Plot Ch2 Waveforms (Peak Focus)
        plot_multi_pmt_waveforms_comparison(
            pmt_ch2_wf_data_for_comparison,
            f"Time Domain Waveforms: Ch2 (Representative Events) - LOM {current_lom_name}",
            f"{lom_output_prefix}all_pmts_time_domain_ch2_comparison", # Base filename
            output_dirs["waveform_analysis_ch2"],
            x_range_mode="peak_focus"
        )

        # Plot Ch2 Waveforms (Broad View)
        plot_multi_pmt_waveforms_comparison(
            pmt_ch2_wf_data_for_comparison,
            f"Time Domain Waveforms: Ch2 (Representative Events) - LOM {current_lom_name}",
            f"{lom_output_prefix}all_pmts_time_domain_ch2_comparison", # Base filename
            output_dirs["waveform_analysis_ch2"],
            x_range_mode="broad"
        )

        # Plot Ch1 FFTs
        plot_multi_pmt_ffts_comparison(
            pmt_ch1_fft_data_for_comparison,
            f"Frequency Domain Spectra: Ch1 (Representative Events) - LOM {current_lom_name}",
            f"{lom_output_prefix}all_pmts_freq_domain_ch1_comparison.pdf",
            output_dirs["fft_analysis_ch1"]
        )

        # Plot Ch2 FFTs
        plot_multi_pmt_ffts_comparison(
            pmt_ch2_fft_data_for_comparison,
            f"Frequency Domain Spectra: Ch2 (Representative Events) - LOM {current_lom_name}",
            f"{lom_output_prefix}all_pmts_freq_domain_ch2_comparison.pdf",
            output_dirs["fft_analysis_ch2"]
        )
        print("\n--- Waveform and FFT Analysis and Comparison Plots Complete ---")

    print("\n========================================================")
    print("=== All LOMs Processed for Waveform and FFT Analysis. ===")
    print("========================================================")