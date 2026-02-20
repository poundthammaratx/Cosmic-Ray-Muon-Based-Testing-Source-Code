import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import glob
import sys
import os
import argparse
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import re
from tqdm import tqdm # For progress bar
from scipy.interpolate import griddata # For interpolation for contour plots
from matplotlib import ticker # Import ticker module

# === START: Path and Import Management ===
script_dir = os.path.dirname(os.path.abspath(__file__))
util_path = os.path.join(script_dir, "util")
if util_path not in sys.path:
    sys.path.append(util_path)

try:
    from hdf_reader import load_hdf_file_as_dict
except ImportError:
    print(f"Error: Could not import 'load_hdf_file_as_dict' from 'hdf_reader.py' in {util_path}.")
    print("Please ensure 'hdf_reader.py' exists in the 'util' directory and contains the function.")
    sys.exit(1)

read_hdffile = load_hdf_file_as_dict
# === END: Path and Import Management ===

# --- Constants ---
# Gain of Channel 1 (High-Gain) in pC per PE.
GAIN_CH1_PC_PER_PE = 0.8

# Max acceptable Gain_Ch2 (Low-Gain) in pC per PE.
MAX_GAIN_CH2_PC_PER_PE = 0.2

# --- Helper Functions ---

def get_timestamps(filename, min_charge_pC=0.0, max_charge_pC=float('inf')):
    """
    Retrieves FPGA timestamps and event indices for events from an HDF file.
    Events are selected if Ch2 charge falls within the specified charge range (in pC).
    Handles missing data gracefully.
    """
    ret_timestamps = []
    ret_eventidx = []

    try:
        data = read_hdffile(filename)
    except Exception as e:
        warnings.warn(f"Error loading HDF file {filename}: {e}. Skipping timestamp retrieval.")
        return [], []

    if not data or "data" not in data or "metadata" not in data:
        warnings.warn(f"Missing 'data' or 'metadata' group in {filename}. Skipping timestamp retrieval.")
        return [], []

    try:
        fpga_time = data["data"]["FPGAtime"][()]
        q_ch2_raw = data["data"]["charge_ch2"][()]
        conversion_ch2 = data["metadata"]["conversion_ch2"][()]
    except KeyError as e:
        warnings.warn(f"Missing expected HDF5 dataset in {filename}: {e}. Skipping timestamp retrieval.")
        return [], []
    except Exception as e:
        warnings.warn(f"An error occurred while accessing data in {filename}: {e}. Skipping timestamp retrieval.")
        return [], []

    if fpga_time.size == 0 or q_ch2_raw.size == 0:
        warnings.warn(f"Empty data arrays found in {filename}. Skipping timestamp retrieval.")
        return [], []

    conversion_factor_pC = (1e-6 * (1/60e6) * 1e12) # Convert ADC units to pC
    q_ch2_pC = q_ch2_raw * (conversion_ch2 * conversion_factor_pC)

    min_len = min(len(fpga_time), len(q_ch2_pC))

    # Filter based on Channel 2 charge only, as requested
    mask = (q_ch2_pC[:min_len] >= min_charge_pC) & (q_ch2_pC[:min_len] <= max_charge_pC)

    ret_timestamps = fpga_time[:min_len][mask].tolist()
    ret_eventidx = np.arange(min_len)[mask].tolist()

    return ret_timestamps, ret_eventidx

def event_matching(timestamp_ref_ns, timestamp_ns, window_ns=100.):
    """
    Finds matching timestamps between a reference set and a target set
    within a specified time window. This function uses a two-pointer approach
    and assumes both input timestamp lists are sorted.
    """
    ngood = 0
    timestamp_ref_ns = np.asarray(timestamp_ref_ns)
    timestamp_ns = np.asarray(timestamp_ns)

    if timestamp_ref_ns.size == 0 or timestamp_ns.size == 0:
        return 0

    ptr_ref, ptr_target = 0, 0
    while ptr_ref < len(timestamp_ref_ns) and ptr_target < len(timestamp_ns):
        t_ref = timestamp_ref_ns[ptr_ref]
        t_target = timestamp_ns[ptr_target]

        if abs(t_ref - t_target) <= window_ns:
            ngood += 1
            ptr_ref += 1
            ptr_target += 1
        elif t_ref > t_target + window_ns:
            ptr_target += 1
        else:
            ptr_ref += 1
    return ngood

def get_gain_ch2_from_equations_summary_file(lom_folder_path, gain_ch1_pc_per_pe, max_gain_ch2_pc_per_pe):
    """
    Reads the 'Fitting Equations Summary' text file for a given LOM, parses each PMT's
    Q2 = Slope * Q1 + Intercept equation, and calculates Gain_Ch2 for each PMT.
    Applies a cap to calculated Gain_Ch2.
    Returns a dictionary: {pmt_id: gain_ch2_pc_per_pe_value}.
    """
    pmt_id_to_gain_ch2 = {}
    lom_name = os.path.basename(os.path.normpath(lom_folder_path))

    summary_file_basename = f"{lom_name}_Run909_charge_correlation_equations_summary.txt"
    summary_file_path = os.path.join(lom_folder_path, summary_file_basename)

    try:
        with open(summary_file_path, 'r') as f:
            for line in f:
                match = re.search(r'PMT (\d{2,3}): Q2 = ([\d.-]+)\s*\*\s*Q1\s*\+\s*([\d.-]+)\s*pC', line)
                if match:
                    pmt_id = int(match.group(1))
                    slope = float(match.group(2))

                    gain_ch2_pc_per_pe = slope * gain_ch1_pc_per_pe

                    if np.isnan(gain_ch2_pc_per_pe) or gain_ch2_pc_per_pe <= 0:
                        warnings.warn(f"Calculated Gain_Ch2 for LOM {lom_name} PMT {pmt_id} is non-positive or NaN from slope {slope}. Setting to NaN.")
                        pmt_id_to_gain_ch2[pmt_id] = np.nan
                    elif gain_ch2_pc_per_pe > max_gain_ch2_pc_per_pe:
                        warnings.warn(f"Calculated Gain_Ch2 for LOM {lom_name} PMT {pmt_id} ({gain_ch2_pc_per_pe:.4f} pC/PE) exceeds MAX_GAIN_CH2_PC_PER_PE ({max_gain_ch2_pc_per_pe:.2f}). Setting to NaN.")
                        pmt_id_to_gain_ch2[pmt_id] = np.nan
                    else:
                        pmt_id_to_gain_ch2[pmt_id] = gain_ch2_pc_per_pe

        if not pmt_id_to_gain_ch2:
            warnings.warn(f"No PMT equations found in {summary_file_path}. Cannot determine Gain_Ch2 for any PMT in this LOM.")
            return None

        return pmt_id_to_gain_ch2

    except FileNotFoundError:
        warnings.warn(f"Equations summary file not found for {lom_name}: {summary_file_path}. Cannot calculate Gain_Ch2 for this LOM.")
        return None
    except Exception as e:
        warnings.warn(f"Error reading or parsing equations summary file {summary_file_path}: {e}. Cannot calculate Gain_Ch2 for this LOM.")
        return None

# --- Refactored Helper: Calculate Coincidence Matrix Data for a single LOM and Threshold ---
def calculate_coincidence_matrix_data(lom_folder_path, fixed_coincidence_window_ns, current_threshold_pe, num_channels):
    current_lom_name = os.path.basename(os.path.normpath(lom_folder_path))

    pmt_id_to_gain_ch2_for_this_lom = get_gain_ch2_from_equations_summary_file(lom_folder_path, GAIN_CH1_PC_PER_PE, MAX_GAIN_CH2_PC_PER_PE)
    if pmt_id_to_gain_ch2_for_this_lom is None:
        warnings.warn(f"Skipping LOM {current_lom_name} for matrix calculation due to missing Gain_Ch2 info.")
        return None, None, None, current_lom_name # Return None for data if gain info is missing

    pmt_to_infile_map = {}
    current_lom_pmt_files = glob.glob(os.path.join(lom_folder_path, "*.hdf"))
    current_lom_pmt_files.sort()
    for fname in current_lom_pmt_files:
        file_base_no_ext = os.path.splitext(os.path.basename(fname))[0]
        pmt_id_match = re.search(r'\.(\d{2,3})$', file_base_no_ext)
        if pmt_id_match:
            pmt_id = int(pmt_id_match.group(1))
            if 0 <= pmt_id < num_channels:
                pmt_to_infile_map[pmt_id] = fname

    pmt_ids_found_in_files = sorted(pmt_to_infile_map.keys())

    if not pmt_ids_found_in_files:
        warnings.warn(f"No HDF files for PMT IDs 0-{num_channels-1} found for LOM {current_lom_name}. Skipping matrix calculation.")
        return None, None, None, current_lom_name

    all_pmt_livetimes = {}
    for pmt_id in pmt_ids_found_in_files:
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

    current_pmt_event_times_filtered = {}
    for pmt_id in range(num_channels):
        infile_pmt = pmt_to_infile_map.get(pmt_id)
        gain_for_this_pmt_ch2 = pmt_id_to_gain_ch2_for_this_lom.get(pmt_id)

        if infile_pmt and gain_for_this_pmt_ch2 is not None and not np.isnan(gain_for_this_pmt_ch2):
            current_threshold_pc_for_this_pmt = current_threshold_pe * gain_for_this_pmt_ch2
            filtered_ts_current_th, _ = get_timestamps(infile_pmt, min_charge_pC=current_threshold_pc_for_this_pmt, max_charge_pC=float('inf'))
            current_pmt_event_times_filtered[pmt_id] = np.sort(np.array(filtered_ts_current_th))
        else:
            current_pmt_event_times_filtered[pmt_id] = np.array([])
            if infile_pmt and (gain_for_this_pmt_ch2 is None or np.isnan(gain_for_this_pmt_ch2)):
                warnings.warn(f"No valid Gain_Ch2 found for LOM {current_lom_name} PMT {pmt_id} or gain capped. This PMT's events will not be filtered by threshold.")

    coincidence_matrix_rates = np.full((num_channels, num_channels), np.nan, dtype=float)

    for i_pmt in range(num_channels):
        for j_pmt in range(num_channels):
            if i_pmt == j_pmt:
                coincidence_matrix_rates[i_pmt, j_pmt] = np.nan
                continue

            if i_pmt < j_pmt: # Calculate for unique pairs only (upper triangle)
                times_i = current_pmt_event_times_filtered.get(i_pmt, np.array([]))
                times_j = current_pmt_event_times_filtered.get(j_pmt, np.array([]))

                count = 0
                if times_i.size > 0 and times_j.size > 0:
                    count = event_matching(times_i, times_j, window_ns=fixed_coincidence_window_ns)

                common_livetime = min(all_pmt_livetimes.get(i_pmt, 0.0), all_pmt_livetimes.get(j_pmt, 0.0))

                rate = 0.0
                if common_livetime > 0:
                    rate = count / common_livetime

                coincidence_matrix_rates[i_pmt, j_pmt] = rate
                coincidence_matrix_rates[j_pmt, i_pmt] = rate # Populate lower triangle symmetrically
    
    # Channel ordering for plotting (consistent for all plots)
    plot_x_channels_order = np.array([14, 12, 10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9, 11, 13, 15])
    plot_y_channels_order = np.array([15, 13, 11, 9, 7, 5, 3, 1, 0, 2, 4, 6, 8, 10, 12, 14])

    ordered_matrix_for_plot = np.zeros((len(plot_y_channels_order), len(plot_x_channels_order)), dtype=float)
    ordered_matrix_for_plot.fill(np.nan)

    for row_idx, pmt_y in enumerate(plot_y_channels_order):
        for col_idx, pmt_x in enumerate(plot_x_channels_order):
            if pmt_x < num_channels and pmt_y < num_channels:
                ordered_matrix_for_plot[row_idx, col_idx] = coincidence_matrix_rates[pmt_y, pmt_x]

    return ordered_matrix_for_plot, pmt_id_to_gain_ch2_for_this_lom, all_pmt_livetimes, current_lom_name


# --- Function for plotting single LOM Coincidence Matrices (Linear Scale) ---
def plot_single_lom_coincidence_matrix_linear(lom_folder_path, output_dir_base, fixed_coincidence_window_ns, current_threshold_pe,
                                         num_channels=16):
    
    ordered_matrix_for_plot, pmt_id_to_gain_ch2_for_this_lom, all_pmt_livetimes, current_lom_name = \
        calculate_coincidence_matrix_data(lom_folder_path, fixed_coincidence_window_ns, current_threshold_pe, num_channels)
    
    if ordered_matrix_for_plot is None:
        return # Skip if data could not be calculated

    print(f"\n--- Generating Coincidence Matrix for LOM: {current_lom_name} @ Threshold: {current_threshold_pe:.1f} PE ---")

    # Ensure output directory for this LOM/Threshold exists
    output_subdir = os.path.join(output_dir_base, f"{current_lom_name}")
    os.makedirs(output_subdir, exist_ok=True)

    # Create a single figure for this threshold
    fig_matrix_single_linear = plt.figure(figsize=(8, 7)) # Adjusted size for a single plot
    ax_matrix_linear = fig_matrix_single_linear.add_subplot(111)

    fig_matrix_single_linear.suptitle(f"LOM {current_lom_name} - Threshold: {current_threshold_pe:.1f} PE\nCoincidence Rate (r'$\Delta T$': {fixed_coincidence_window_ns} ns) [Hz]", fontsize=14, y=0.98)

    selected_cmap = plt.cm.get_cmap('Blues')

    # Channel ordering for plotting (reused from calculate_coincidence_matrix_data)
    plot_x_channels_order = np.array([14, 12, 10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9, 11, 13, 15])
    plot_y_channels_order = np.array([15, 13, 11, 9, 7, 5, 3, 1, 0, 2, 4, 6, 8, 10, 12, 14])

    # For linear scale, simply use the data, nan values will be masked by pcolormesh
    # No fixed vmin/vmax, let matplotlib determine automatically
    coincidence_matrix_rates_ordered_linear = np.ma.masked_where(np.isnan(ordered_matrix_for_plot), ordered_matrix_for_plot)

    mesh = ax_matrix_linear.pcolormesh(np.arange(len(plot_x_channels_order)),
                                     np.arange(len(plot_y_channels_order)),
                                     coincidence_matrix_rates_ordered_linear,
                                     cmap=selected_cmap,
                                     edgecolors='lightgray',
                                     linewidth=0.5) # Removed vmin/vmax

    ax_matrix_linear.set_xticks(np.arange(len(plot_x_channels_order)) + 0.5)
    ax_matrix_linear.set_yticks(np.arange(len(plot_y_channels_order)) + 0.5)
    ax_matrix_linear.set_xticklabels(plot_x_channels_order, fontsize=12) # Adjusted fontsize
    ax_matrix_linear.set_yticklabels(plot_y_channels_order, fontsize=12) # Adjusted fontsize

    ax_matrix_linear.set_xlabel('PMT ID', fontsize=15)
    ax_matrix_linear.set_ylabel('PMT ID', fontsize=15)

    ax_matrix_linear.invert_yaxis()
    ax_matrix_linear.set_aspect('equal', adjustable='box')

    # Lines at PMT No. 0 for X-axis and PMT No. 0 for Y-axis
    try:
        split_idx_x_for_0 = np.where(plot_x_channels_order == 0)[0][0]
        split_idx_y_for_0 = np.where(plot_y_channels_order == 1)[0][0]
        ax_matrix_linear.axvline(split_idx_x_for_0 + 0.5, color='black', linestyle='-', linewidth=1.5)
        ax_matrix_linear.axhline(split_idx_y_for_0 + 0.5, color='black', linestyle='-', linewidth=1.5)
    except IndexError:
        pass

    # Add a single colorbar for the entire figure
    # --- Colorbar Customization ---
    # Let matplotlib determine ticks and formatting automatically
    cbar = fig_matrix_single_linear.colorbar(mesh, ax=ax_matrix_linear,
                                           shrink=0.7, pad=0.03) # No explicit ticks or format

    cbar.ax.tick_params(labelsize=9) # Adjusted font size of colorbar labels to be smaller
    cbar.set_label("Coincidence rate (r'$\Delta T$' = 100 ns) [Hz]", rotation=270, labelpad=15, fontsize=12)

    fig_matrix_single_linear.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])

    matrices_linear_filename = os.path.join(output_subdir, f"{current_lom_name}_coincidence_matrix_Threshold_{current_threshold_pe:.1f}PE_linear.pdf")
    fig_matrix_single_linear.savefig(matrices_linear_filename)
    plt.close(fig_matrix_single_linear)
    print(f'Saved linear coincidence matrix for LOM {current_lom_name} @ Threshold {current_threshold_pe:.1f} PE to: {matrices_linear_filename}')

    # --- Save PMT Gain_Ch2 Data for this LOM ---
    if pmt_id_to_gain_ch2_for_this_lom:
        gain_data_for_df = [{"LOM": current_lom_name, "PMT_ID": pmt_id, "Gain_Ch2_pC_per_PE": gain_value}
                            for pmt_id, gain_value in pmt_id_to_gain_ch2_for_this_lom.items()]
        gain_df = pd.DataFrame(gain_data_for_df)
        gain_df_sorted = gain_df.sort_values(by="PMT_ID").reset_index(drop=True)

        # Save to the LOM-specific subdirectory
        excel_filename = os.path.join(output_subdir, f"{current_lom_name}_PMT_Gain_Ch2_Summary.xlsx")
        try:
            gain_df_sorted.to_excel(excel_filename, index=False, engine='openpyxl')
            if len(COINCIDENCE_THRESHOLDS_PE) > 0 and current_threshold_pe == COINCIDENCE_THRESHOLDS_PE[0]:
                print(f"Saved PMT Gain_Ch2 summary for {current_lom_name} to: {excel_filename}")
        except ImportError:
            warnings.warn("To save to .xlsx, please install 'openpyxl' (pip install openpyxl). Skipping Excel export.")
        except Exception as e:
            warnings.warn(f"Error saving to Excel for {current_lom_name}: {e}. Skipping Excel export.")
    else:
        if len(COINCIDENCE_THRESHOLDS_PE) > 0 and current_threshold_pe == COINCIDENCE_THRESHOLDS_PE[0]:
            print(f"No PMT Gain_Ch2 data collected for LOM {current_lom_name}.")

# --- NEW Function: Plot Coincidence Matrices for Multiple LOMs on one page (comparison, Linear Scale) ---
def plot_multi_lom_coincidence_matrices_comparison_linear(lom_data_dirs, output_dir_base, fixed_coincidence_window_ns, current_threshold_pe, num_channels=16):
    print(f"\n--- Generating Multi-LOM Coincidence Comparison Plot @ Threshold: {current_threshold_pe:.1f} PE ---")

    # Define subplot grid (e.g., 2 rows, 3 columns for 6 LOMs)
    n_rows = 2
    n_cols = 3
    
    # Create the figure and subplots
    fig_comparison, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.flatten() # Flatten the 2D array of axes for easy iteration

    fig_comparison.suptitle(f"Coincidence Rate Matrices Comparison (r'$\Delta T$': {fixed_coincidence_window_ns} ns) [Hz]\nThreshold: {current_threshold_pe:.1f} PE", fontsize=16, y=0.98)

    selected_cmap = plt.cm.get_cmap('Blues')

    # Channel ordering for plotting (consistent for all plots)
    plot_x_channels_order = np.array([14, 12, 10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9, 11, 13, 15])
    plot_y_channels_order = np.array([15, 13, 11, 9, 7, 5, 3, 1, 0, 2, 4, 6, 8, 10, 12, 14])

    # Filter out non-existing LOM directories and sort them for consistent plotting order
    existing_lom_dirs = [d for d in lom_data_dirs if os.path.isdir(d)]
    sorted_lom_dirs = sorted(existing_lom_dirs)

    if not sorted_lom_dirs:
        warnings.warn("No valid LOM data directories found for comparison plot. Skipping.")
        plt.close(fig_comparison)
        return
        
    if len(sorted_lom_dirs) > n_rows * n_cols:
        warnings.warn(f"Too many LOM directories ({len(sorted_lom_dirs)}) for the configured {n_rows}x{n_cols} subplot grid. Plotting only the first {n_rows*n_cols} LOMs.")
        sorted_lom_dirs = sorted_lom_dirs[:n_rows * n_cols]

    mappables = [] # To store mappables for the shared colorbar

    for i, lom_dir in enumerate(tqdm(sorted(sorted_lom_dirs), desc=f"Plotting LOMs for {current_threshold_pe:.1f} PE")):
        ax = axes[i]
        
        ordered_matrix_for_plot, _, _, current_lom_name = \
            calculate_coincidence_matrix_data(lom_dir, fixed_coincidence_window_ns, current_threshold_pe, num_channels)
        
        if ordered_matrix_for_plot is None:
            ax.text(0.5, 0.5, 'Data N/A', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title(f"LOM {os.path.basename(os.path.normpath(lom_dir))}", fontsize=10)
            # Ensure ticks and labels are set even if no data, for consistent layout
            ax.set_xticks(np.arange(len(plot_x_channels_order)) + 0.5)
            ax.set_xticklabels(plot_x_channels_order, fontsize=10)
            ax.set_yticks(np.arange(len(plot_y_channels_order)) + 0.5)
            ax.set_yticklabels(plot_y_channels_order, fontsize=10)
            ax.invert_yaxis()
            ax.set_aspect('equal', adjustable='box')
            # Add black lines even if no data, for structure consistency
            try:
                split_idx_x_for_0 = np.where(plot_x_channels_order == 0)[0][0]
                split_idx_y_for_0 = np.where(plot_y_channels_order == 1)[0][0]
                ax.axvline(split_idx_x_for_0 + 0.5, color='black', linestyle='-', linewidth=1.0)
                ax.axhline(split_idx_y_for_0 + 0.5, color='black', linestyle='-', linewidth=1.0)
            except IndexError:
                pass
            continue

        # For linear scale, simply use the data, nan values will be masked by pcolormesh
        # No fixed vmin/vmax, let matplotlib determine automatically
        coincidence_matrix_rates_ordered_linear = np.ma.masked_where(np.isnan(ordered_matrix_for_plot), ordered_matrix_for_plot)

        mesh = ax.pcolormesh(np.arange(len(plot_x_channels_order)),
                                     np.arange(len(plot_y_channels_order)),
                                     coincidence_matrix_rates_ordered_linear,
                                     cmap=selected_cmap,
                                     edgecolors='lightgray',
                                     linewidth=0.5) # Removed vmin/vmax
        mappables.append(mesh)

        ax.set_xticks(np.arange(len(plot_x_channels_order)) + 0.5)
        ax.set_yticks(np.arange(len(plot_y_channels_order)) + 0.5)
        ax.set_xticklabels(plot_x_channels_order, fontsize=10)
        ax.set_yticklabels(plot_y_channels_order, fontsize=10)
        ax.set_title(f"LOM {current_lom_name}", fontsize=10) # Set title as LOM name
        
        # Set labels only for bottom row and leftmost column
        if i >= (n_rows - 1) * n_cols: # If in the last row
            ax.set_xlabel('PMT ID', fontsize=12)
        if i % n_cols == 0: # If in the first column
            ax.set_ylabel('PMT ID', fontsize=12)

        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')

        # Lines at PMT No. 0 for X-axis and PMT No. 0 for Y-axis
        try:
            split_idx_x_for_0 = np.where(plot_x_channels_order == 0)[0][0]
            split_idx_y_for_0 = np.where(plot_y_channels_order == 1)[0][0]
            ax.axvline(split_idx_x_for_0 + 0.5, color='black', linestyle='-', linewidth=1.0)
            ax.axhline(split_idx_y_for_0 + 0.5, color='black', linestyle='-', linewidth=1.0)
        except IndexError:
            pass

    # Hide unused subplots if there are fewer LOMs than subplots
    for i in range(len(sorted_lom_dirs), n_rows * n_cols):
        fig_comparison.delaxes(axes[i])

    # Add a single colorbar for the entire figure (if there are mappables)
    if mappables:
        # Let matplotlib determine ticks and formatting automatically
        # Create a new axes specifically for the colorbar at the right of the figure
        cbar_ax = fig_comparison.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height] in figure coordinates

        cbar = fig_comparison.colorbar(mappables[0], cax=cbar_ax) # No explicit ticks or format

        cbar.ax.tick_params(labelsize=9) # Adjusted font size
        cbar.set_label("Coincidence rate (r'$\Delta T$' = 100 ns) [Hz]", rotation=270, labelpad=15, fontsize=12)

    fig_comparison.tight_layout(rect=[0.02, 0.02, 0.90, 0.95]) # Adjust rect to leave space for the manually placed colorbar

    comparison_filename = os.path.join(output_dir_base, f"MultiLOM_Coincidence_Comparison_Threshold_{current_threshold_pe:.1f}PE_linear.pdf")
    fig_comparison.savefig(comparison_filename)
    plt.close(fig_comparison)
    print(f'Saved multi-LOM comparison plot @ Threshold {current_threshold_pe:.1f} PE to: {comparison_filename}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Coincidence Rate Matrices (Linear Scale) for All LOMs.")
    parser.add_argument("lom_data_dirs", nargs='+', help="Input LOM data directories (e.g., 'path/to/LOM16-01/'). Each directory should contain HDF files for PMT channels and the equations summary TXT file.")
    parser.add_argument("--output_dir", "-o", help="Output directory for plots (default: ./plots_output)", type=str, default='./plots_output')
    parser.add_argument("--num_channels", type=int, default=16, help="Total number of PMT channels (e.g., 0-15).")
    parser.add_argument("--plot_contours", action="store_true", help="Set this flag to plot contour lines over the coincidence matrix (Note: contour plotting logic is currently disabled in the function).")

    args = parser.parse_args()

    # --- Configuration Parameters ---
    FIXED_COINCIDENCE_WINDOW_NS = 100.0
    COINCIDENCE_THRESHOLDS_PE = np.array([10.0, 20.0, 50.0, 100.0, 200.0]) # Your current thresholds

    output_base_dir = args.output_dir
    os.makedirs(output_base_dir, exist_ok=True)

    print(f"Output plots and data will be saved to: {output_base_dir}")

    # --- Loop through each LOM directory and then each threshold for individual plots ---
    existing_lom_dirs = [d for d in args.lom_data_dirs if os.path.isdir(d)]
    sorted_lom_dirs = sorted(existing_lom_dirs)

    print("\nStarting individual LOM plots (Linear Scale, Auto Range)...")
    for lom_dir in sorted_lom_dirs:
        for threshold_pe in COINCIDENCE_THRESHOLDS_PE:
            plot_single_lom_coincidence_matrix_linear(
                lom_dir,
                output_base_dir,
                FIXED_COINCIDENCE_WINDOW_NS,
                threshold_pe,
                num_channels=args.num_channels
            )

    print("\nStarting multi-LOM comparison plots (Linear Scale, Auto Range)...")
    # --- Loop through each threshold for multi-LOM comparison plots ---
    for threshold_pe in COINCIDENCE_THRESHOLDS_PE:
        plot_multi_lom_coincidence_matrices_comparison_linear(
            args.lom_data_dirs,
            output_base_dir,
            FIXED_COINCIDENCE_WINDOW_NS,
            threshold_pe,
            num_channels=args.num_channels
        )

    print("\n========================================================")
    print("=== All Coincidence Matrices Plots Complete (Linear, Auto Range). ===")
    print("========================================================")