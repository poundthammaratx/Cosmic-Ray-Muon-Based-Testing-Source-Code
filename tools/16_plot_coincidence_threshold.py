import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore") # Suppress warnings, use with caution in production
import pandas as pd
import glob
import sys
import os
import argparse
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import re
from tqdm import tqdm # For progress bar
from scipy.stats import gaussian_kde # For smoothing histograms
from scipy.interpolate import interp1d # For smoothing the lines

# === START: Path and Import Management ===
# This part assumes 'hdf_reader.py' is located in a 'util' sub-directory
script_dir = os.path.dirname(os.path.abspath(__file__))
util_path = os.path.join(script_dir, "util")
if util_path not in sys.path:
    sys.path.append(util_path) # Corrected: removed extra 'sys.'

try:
    # Attempt to import the required function from hdf_reader.py
    from hdf_reader import load_hdf_file_as_dict
except ImportError:
    print(f"Error: Could not import 'load_hdf_file_as_dict' from 'hdf_reader.py' in {util_path}.")
    print("Please ensure 'hdf_reader.py' exists in the 'util' directory and contains the function.")
    print("Example: Your main script -> your_script.py, then create a folder 'util' in the same directory,")
    print("and place 'hdf_reader.py' inside 'util'.")
    sys.exit(1)

read_hdffile = load_hdf_file_as_dict
# === END: Path and Import Management ===

# --- Constants ---
# Gain of Channel 1 (High-Gain) in pC per PE.
GAIN_CH1_PC_PER_PE = 0.8

# Max acceptable Gain_Ch2 (Low-Gain) in pC per PE.
MAX_GAIN_CH2_PC_PER_PE = 0.2

# Dead time for correction in microseconds (example value, adjust as needed)
DEAD_TIME_MU_S = 1.0 # Use 1.0 microsecond as example, user can adjust

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

    # Filter based on Channel 2 charge only
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
    Splits the summary file name dynamically based on LOM name.
    Applies a cap to calculated Gain_Ch2.
    Returns a dictionary: {pmt_id: gain_ch2_pc_per_pe_value}.
    """
    pmt_id_to_gain_ch2 = {}
    lom_name = os.path.basename(os.path.normpath(lom_folder_path))

    # Dynamically build summary file name, assuming it's {LOM_NAME}_Run909_charge_correlation_equations_summary.txt
    summary_file_basename = f"{lom_name}_Run909_charge_correlation_equations_summary.txt"
    summary_file_path = os.path.join(lom_folder_path, summary_file_basename)

    try:
        with open(summary_file_path, 'r') as f:
            for line in f:
                # Updated regex to be more flexible with the LOM name prefix in the line
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

# --- Main Plotting Function for Coincidence Rate vs Threshold ---

def plot_coincidence_rate_vs_threshold(lom_data_dirs, output_dir_base, fixed_coincidence_window_ns, coincidence_thresholds_pe,
                                        num_channels, selected_pmt_pairs_with_markers, dead_time_mu_s, output_filename_suffix="",
                                        geant4_simulation_data=None): # Added geant4_simulation_data argument

    # Convert dead time from microseconds to seconds
    dead_time_s = dead_time_mu_s * 1e-6

    # Filter out LOM directories that don't exist
    existing_lom_dirs = [d for d in lom_data_dirs if os.path.isdir(d)]
    if not existing_lom_dirs:
        print("No valid LOM data directories found. Exiting plotting function.")
        return

    all_coincidence_data = []

    for lom_idx, current_lom_folder_path in enumerate(tqdm(sorted(existing_lom_dirs), desc="Processing LOMs")):
        current_lom_folder_path = os.path.normpath(current_lom_folder_path)
        current_lom_name = os.path.basename(current_lom_folder_path)

        # --- DEBUG PRINT ---
        print(f"\n--- LOM: {current_lom_name} ---")

        pmt_id_to_gain_ch2_for_this_lom = get_gain_ch2_from_equations_summary_file(current_lom_folder_path, GAIN_CH1_PC_PER_PE, MAX_GAIN_CH2_PC_PER_PE)
        # --- DEBUG PRINT ---
        print(f"PMT ID to Gain_Ch2 (for this LOM): {pmt_id_to_gain_ch2_for_this_lom}")

        if pmt_id_to_gain_ch2_for_this_lom is None:
            warnings.warn(f"Skipping LOM {current_lom_name} due to inability to determine valid Gain_Ch2 for any PMT from summary file.")
            continue

        pmt_to_infile_map = {}
        current_lom_pmt_files = glob.glob(os.path.join(current_lom_folder_path, "*.hdf"))
        current_lom_pmt_files.sort()
        for fname in current_lom_pmt_files:
            file_basename = os.path.basename(fname)
            pmt_id_match = re.search(r'\.(\d{2,3})\.hdf$', file_basename)
            if pmt_id_match:
                pmt_id = int(pmt_id_match.group(1))
                if 0 <= pmt_id < num_channels: # Ensure PMT ID is within expected range
                    pmt_to_infile_map[pmt_id] = fname
                else:
                    warnings.warn(f"Found HDF file {file_basename} with PMT ID {pmt_id}, but it's outside expected range (0-{num_channels-1}). Skipping.")
            else:
                warnings.warn(f"Could not extract PMT ID from filename {file_basename}. Skipping.")

        # --- DEBUG PRINT ---
        print(f"PMT IDs found in files for {current_lom_name}: {sorted(pmt_to_infile_map.keys())}")

        pmt_ids_found_in_files = sorted(pmt_to_infile_map.keys())

        if not pmt_ids_found_in_files:
            warnings.warn(f"No HDF files for PMT IDs 0-{num_channels-1} found for LOM {current_lom_name}. Skipping this LOM in plot.")
            continue

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
            # --- DEBUG PRINT ---
            print(f"    Livetime for PMT {pmt_id} from {os.path.basename(infile_pmt)}: {pmt_livetime:.2f} s (FPGAtime size: {all_raw_fpga_times.size})")


        for current_threshold_pe in tqdm(coincidence_thresholds_pe, desc=f"    LOM {current_lom_name} Thresholds", leave=False):
            for (pmt1_id, pmt2_id), marker in selected_pmt_pairs_with_markers.items():
                gain_pmt1_ch2 = pmt_id_to_gain_ch2_for_this_lom.get(pmt1_id)
                gain_pmt2_ch2 = pmt_id_to_gain_ch2_for_this_lom.get(pmt2_id)

                infile_pmt1 = pmt_to_infile_map.get(pmt1_id)
                infile_pmt2 = pmt_to_infile_map.get(pmt2_id)

                # Initialize with NaN
                measured_rate = np.nan
                coincidence_count = 0
                rate_error = np.nan
                true_rate = np.nan # Still calculate true_rate for internal use, just don't plot it directly

                if infile_pmt1 and infile_pmt2 and \
                   gain_pmt1_ch2 is not None and not np.isnan(gain_pmt1_ch2) and \
                   gain_pmt2_ch2 is not None and not np.isnan(gain_pmt2_ch2):

                    # Convert PE threshold to pC for each PMT using its specific Ch2 gain
                    threshold_pc_pmt1 = current_threshold_pe * gain_pmt1_ch2
                    threshold_pc_pmt2 = current_threshold_pe * gain_pmt2_ch2

                    times_pmt1, _ = get_timestamps(infile_pmt1, min_charge_pC=threshold_pc_pmt1, max_charge_pC=float('inf'))
                    times_pmt2, _ = get_timestamps(infile_pmt2, min_charge_pC=threshold_pc_pmt2, max_charge_pC=float('inf'))

                    coincidence_count = event_matching(times_pmt1, times_pmt2, window_ns=fixed_coincidence_window_ns)

                    common_livetime = min(all_pmt_livetimes.get(pmt1_id, 0.0), all_pmt_livetimes.get(pmt2_id, 0.0))

                    if common_livetime > 0:
                        measured_rate = coincidence_count / common_livetime

                        # Task 1: Calculate Error (Poisson statistics)
                        if coincidence_count > 0:
                            rate_error = np.sqrt(coincidence_count) / common_livetime
                        else:
                            rate_error = 0.0 # If no counts, error is 0

                        # Task 2: Apply Dead Time Correction (Keep calculation, but won't be explicitly plotted unless user changes mind)
                        # Ensure (1 - measured_rate * dead_time_s) is not zero or negative
                        denominator = (1 - measured_rate * dead_time_s)
                        if denominator > 0:
                            true_rate = measured_rate / denominator
                        else:
                            true_rate = np.nan # Undefined if R_measured * tau >= 1
                            warnings.warn(f"Dead time correction resulted in non-positive denominator for LOM {current_lom_name} PMT Pair {pmt1_id}-{pmt2_id} @ Threshold {current_threshold_pe} PE (Measured Rate: {measured_rate:.4f} Hz, Dead Time: {dead_time_mu_s} us).")

                    else:
                        measured_rate = 0.0
                        rate_error = 0.0
                        true_rate = 0.0 # If no common livetime, rate is 0

                    # --- DEBUG PRINT ---
                    print(f"      PMT Pair {pmt1_id}-{pmt2_id} @ Threshold {current_threshold_pe} PE:")
                    print(f"        Gain Ch2 PMT1: {gain_pmt1_ch2:.4f} pC/PE, Gain Ch2 PMT2: {gain_pmt2_ch2:.4f} pC/PE")
                    print(f"        Threshold pC PMT1: {threshold_pc_pmt1:.4f} pC, Threshold pC PMT2: {threshold_pc_pmt2:.4f} pC")
                    print(f"        Times PMT1 len: {len(times_pmt1)}, Times PMT2 len: {len(times_pmt2)}")
                    print(f"        Coincidence Count: {coincidence_count}, Common Livetime: {common_livetime:.2f} s")
                    print(f"        Measured Rate: {measured_rate:.4f} Hz, Error: {rate_error:.4f} Hz, True Rate (Dead Time Corrected): {true_rate:.4f} Hz")

                else:
                    # Issue a warning if a file or valid gain is missing for a pair
                    # --- DEBUG PRINT ---
                    print(f"      PMT Pair {pmt1_id}-{pmt2_id} @ Threshold {current_threshold_pe} PE: Data/Gain missing or invalid, Rates set to NaN.")
                    if not infile_pmt1: warnings.warn(f"Missing HDF file for LOM {current_lom_name} PMT {pmt1_id}. Cannot calculate coincidence rate.")
                    if not infile_pmt2: warnings.warn(f"Missing HDF file for LOM {current_lom_name} PMT {pmt2_id}. Cannot calculate coincidence rate.")
                    if gain_pmt1_ch2 is None or np.isnan(gain_pmt1_ch2): warnings.warn(f"No valid Gain_Ch2 for LOM {current_lom_name} PMT {pmt1_id}.")
                    if gain_pmt2_ch2 is None or np.isnan(gain_pmt2_ch2): warnings.warn(f"No valid Gain_Ch2 for LOM {current_lom_name} PMT {pmt2_id}.")

                all_coincidence_data.append({
                    "LOM": current_lom_name,
                    "PMT_Pair": f"{pmt1_id}-{pmt2_id}",
                    "Threshold_PE": current_threshold_pe,
                    "Coincidence_Count": coincidence_count, # Store count
                    "Common_Livetime_s": common_livetime, # Store livetime
                    "Coincidence_Rate_Measured_Hz": measured_rate, # Store measured rate
                    "Coincidence_Rate_Error_Hz": rate_error, # Store error
                    "Coincidence_Rate_True_Hz": true_rate # Store true rate (dead time corrected)
                })

    if not all_coincidence_data:
        print("No coincidence data collected to plot. Exiting.")
        return

    df_coincidence = pd.DataFrame(all_coincidence_data)
    # --- DEBUG PRINT ---
    print("\n--- Final Coincidence DataFrame (first 20 rows): ---")
    print(df_coincidence.head(20))
    print("\n--- Coincidence Rate Statistics: ---")
    print(df_coincidence['Coincidence_Rate_True_Hz'].describe()) # Use True Rate for stats
    print("\n--- Unique Coincidence Rates (True) found: ---")
    print(df_coincidence['Coincidence_Rate_True_Hz'].unique())

    # --- Save df_coincidence to CSV ---
    csv_filename = os.path.join(output_dir_base, f"Coincidence_Rate_Data{output_filename_suffix}.csv")
    df_coincidence.to_csv(csv_filename, index=False)
    print(f"Saved Coincidence Rate Data to: {csv_filename}")

    # --- Save df_coincidence to XLSX ---
    xlsx_filename = os.path.join(output_dir_base, f"Coincidence_Rate_Data{output_filename_suffix}.xlsx")
    try:
        df_coincidence.to_excel(xlsx_filename, index=False)
        print(f"Saved Coincidence Rate Data to: {xlsx_filename}")
    except ImportError:
        warnings.warn("'openpyxl' is not installed. Cannot save to .xlsx. Please install with: pip install openpyxl")
    except Exception as e:
        warnings.warn(f"Error saving to .xlsx file {xlsx_filename}: {e}")


    # Create the plot
    num_rows = len(df_coincidence['LOM'].unique())
    fig_width_inches = 8.5
    fig_height_per_lom_subplot = 4.0
    fig_height_inches = fig_height_per_lom_subplot * num_rows + 1.5

    # MODIFICATION 1: Removed sharex=True and sharey=True
    fig, axes = plt.subplots(num_rows, 1, figsize=(fig_width_inches, fig_height_inches))

    if num_rows == 1:
        axes = [axes]

    # Adjust title (Dead Time info is removed as per Yuya's latest feedback)
    fig.suptitle(f"Coincidence Rate vs. Threshold (Fixed $\\Delta T$: {fixed_coincidence_window_ns} ns)", fontsize=14, y=0.99)

    # Define x and y axis limits and ticks
    x_ticks = np.arange(20, 161, 20) # X-axis: 20-160, step 20 (as per provided image)

    y_ticks_min = 0.00
    y_ticks_max = 0.30 # Max Y for Rate
    y_ticks_step = 0.05 # Y-axis: 0.00-0.30, step 0.05
    y_ticks = np.arange(y_ticks_min, y_ticks_max + y_ticks_step, y_ticks_step)

    # Loop through each LOM and plot
    for i, lom_name in enumerate(sorted(df_coincidence['LOM'].unique(), reverse=False)):
        ax = axes[i]
        lom_df = df_coincidence[df_coincidence['LOM'] == lom_name]

        print(f"\n--- Plotting for LOM: {lom_name} ---")

        for (pmt1_id, pmt2_id), marker in selected_pmt_pairs_with_markers.items():
            # Adjust Legend Label format
            if (pmt1_id, pmt2_id) == (14, 15):
                pair_label_for_legend = "PMT 14 & 15"
            elif (pmt1_id, pmt2_id) == (12, 13):
                pair_label_for_legend = "PMT 12 & 13"
            elif (pmt1_id, pmt2_id) == (10, 11):
                pair_label_for_legend = "PMT 10 & 11"
            elif (pmt1_id, pmt2_id) == (8, 9):
                pair_label_for_legend = "PMT 09 & 08"
            else:
                if pmt1_id > pmt2_id:
                    pair_label_for_legend = f"PMT {pmt2_id:02d} & {pmt1_id:02d}"
                else:
                    pair_label_for_legend = f"PMT {pmt1_id:02d} & {pmt2_id:02d}"

            pair_label_for_data_filter = f"{pmt1_id}-{pmt2_id}"

            pair_df = lom_df[lom_df['PMT_Pair'] == pair_label_for_data_filter].sort_values(by="Threshold_PE")

            print(f"      PMT Pair: {pair_label_for_data_filter} (Legend: {pair_label_for_legend})")
            print(f"      pair_df is empty: {pair_df.empty}")

            # Data for plotting (using measured rates - no deadtime correction)
            rates_to_plot = []
            thresholds_to_plot = []
            errors_to_plot = []
            if not pair_df.empty:
                rates_to_plot = pair_df['Coincidence_Rate_Measured_Hz'].values # Use Measured Rate
                thresholds_to_plot = pair_df['Threshold_PE'].values
                errors_to_plot = pair_df['Coincidence_Rate_Error_Hz'].values

                print(f"      Number of data points for plotting: {len(rates_to_plot)}")
                print(f"      Rates (Measured) to plot (first 5): {rates_to_plot[:5]}")
                print(f"      Errors to plot (first 5): {errors_to_plot[:5]}")
                print(f"      Thresholds to plot (first 5): {thresholds_to_plot[:5]}")
                print(f"      Rates (Measured) array contains any NaNs: {np.isnan(rates_to_plot).any()}")
            else:
                print(f"      Number of data points for plotting: 0 (pair_df was empty)")
                print(f"      Rates array contains any NaNs: N/A (pair_df was empty)")

            # Plot error bars (no line yet)
            if not pair_df.empty:
                print(f"      --> PLOTTING data for {pair_label_for_data_filter} in {lom_name} <--" )
                ax.errorbar(thresholds_to_plot, rates_to_plot, yerr=errors_to_plot,
                            marker='.', linestyle='None', label=pair_label_for_legend, capsize=3, markersize=3) # Changed marker to '.' and markersize to 3

                # Add smoothing line for PMT pairs (linear interpolation for straight lines)
                if len(thresholds_to_plot) > 1 and not np.any(np.isnan(rates_to_plot)):
                    # Ensure thresholds are sorted for interp1d
                    sorted_indices = np.argsort(thresholds_to_plot)
                    x_sorted = thresholds_to_plot[sorted_indices]
                    y_sorted = rates_to_plot[sorted_indices]

                    # Use interp1d with kind='linear' for straight lines
                    f_interp = interp1d(x_sorted, y_sorted, kind='linear', fill_value="extrapolate")
                    x_new = np.linspace(x_sorted.min(), x_sorted.max(), 500) # Create more points for a smooth line

                    # Get the color of the current plotted errorbar to match the line color
                    line_color = ax.lines[-1].get_color()
                    ax.plot(x_new, f_interp(x_new), color=line_color, linestyle='-')
                else: # Fallback for cases with too few points or NaNs (plot original points with straight lines)
                    ax.plot(thresholds_to_plot, rates_to_plot, marker='.', linestyle='-', color=ax.lines[-1].get_color()) # Changed marker to '.'
            else:
                print(f"      !!! WARNING: Data for LOM {lom_name}, PMT Pair {pair_label_for_data_filter} NOT PLOTTED because pair_df is empty.")

        # === Plot Geant4 Simulation Data ===
        # This section will now use the 'geant4_simulation_data' passed as an argument.
        # This assumes 'geant4_simulation_data' is a dictionary with 'x', 'y', and 'yerr' arrays.
        if geant4_simulation_data is not None and \
           isinstance(geant4_simulation_data, dict) and \
           'x' in geant4_simulation_data and \
           'y' in geant4_simulation_data and \
           'yerr' in geant4_simulation_data:

            g4_x = np.array(geant4_simulation_data['x'])
            g4_y = np.array(geant4_simulation_data['y'])
            g4_yerr = np.array(geant4_simulation_data['yerr'])

            # Filter Geant4 data to the plot's X-axis range (20-150)
            mask_g4 = (g4_x >= 20) & (g4_x <= 150)

            if np.any(mask_g4):
                # Plot error bars (no line yet)
                ax.errorbar(g4_x[mask_g4], g4_y[mask_g4],
                            yerr=g4_yerr[mask_g4],
                            color='black', linestyle='None', marker='.', label='Geant4 simulation', capsize=3, markersize=3) # Changed marker to '.' and markersize to 3

                # Add smoothing line for Geant4 data (linear interpolation for a "straighter" smooth look)
                x_g4_filtered = g4_x[mask_g4]
                y_g4_filtered = g4_y[mask_g4]
                # Ensure x data is sorted for interpolation (should already be sorted but good practice)
                sorted_indices_g4 = np.argsort(x_g4_filtered)
                x_g4_sorted = x_g4_filtered[sorted_indices_g4]
                y_g4_sorted = y_g4_filtered[sorted_indices_g4]

                if len(x_g4_sorted) > 1 and not np.any(np.isnan(y_g4_sorted)):
                    # Use interp1d with kind='linear' for straight lines
                    f_g4_interp = interp1d(x_g4_sorted, y_g4_sorted, kind='linear', fill_value="extrapolate")
                    x_new_g4 = np.linspace(x_g4_sorted.min(), x_g4_sorted.max(), 500)
                    ax.plot(x_new_g4, f_g4_interp(x_new_g4), color='black', linestyle='-')
                else: # Fallback for Geant4 data with too few points or NaNs
                    ax.plot(x_g4_filtered, y_g4_filtered, color='black', linestyle='-', marker='.') # Changed marker to '.'

                warnings.warn("Plotted Geant4 simulation data with smoothing.")
            else:
                warnings.warn("Geant4 simulation data is outside the plot's X-axis range (20-150) or is empty after filtering. Not plotting.")
        else:
            warnings.warn("Geant4 simulation data not provided or is in an invalid format. Not plotting 'Geant4 simulation' line.")


        ax.set_title(f"LOM {lom_name}", fontsize=12)

        # Ensure X-axis limits and ticks are set for each subplot
        ax.set_xticks(x_ticks)
        ax.set_xlim(x_ticks.min(), x_ticks.max())
        ax.tick_params(axis='x', labelsize=10) # Adjust label size if needed
        # MODIFICATION 2: Force X-axis labels to be visible on every subplot
        ax.tick_params(labelbottom=True)


        # Ensure Y-axis limits and ticks are set for each subplot (as they are not shared anymore)
        ax.set_yticks(y_ticks)
        ax.set_ylim(y_ticks.min(), y_ticks.max())
        ax.tick_params(axis='y', labelsize=10) # Adjust label size if needed
        
        # Set x and y labels for every subplot
        ax.set_xlabel("Threshold [PE]", fontsize=10)
        ax.set_ylabel("Cocincidence Rate [1/s]", fontsize=10)


        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='upper right', fontsize=8)

    # Set common labels for shared axes (now explicitly set for each, so these are just for clarity)
    # You might want to remove these if the individual labels suffice
    # axes[-1].set_xlabel("Threshold [PE]", fontsize=10) # REMOVED: Now set per subplot
    # Since y-labels are now per-subplot, we can remove this or set it for the first one only.
    # For consistent look with the example, we might want to put it only on the first subplot
    # or keep it as is, which means it will be slightly off-center if there are many subplots.
    # I'll leave it as is for now, as it indicates the general y-axis meaning.
    # middle_ax_idx = num_rows // 2 if num_rows > 0 else 0 # REMOVED: Now set per subplot
    # axes[middle_ax_idx].set_ylabel("Rate [1/s]", fontsize=10) # REMOVED: Now set per subplot

    # Use plt.tight_layout to prevent labels from overlapping
    plt.tight_layout(rect=[0.05, 0.05, 0.98, 0.95]) # Adjust rect if labels still overlap

    plot_filename = os.path.join(output_dir_base, f"Coincidence_Rate_vs_Threshold{output_filename_suffix}.pdf")
    fig.savefig(plot_filename)
    plt.close(fig)
    print(f'Saved Coincidence Rate vs. Threshold plot to: {plot_filename}')


# --- New Function for Histogram of Time Intervals ---
# No changes needed here, as the x-ticks were already explicitly set per subplot.
def plot_time_interval_histograms(lom_data_dirs, num_channels, output_dir_base):
    print("\n--- Generating Time Interval Histograms ---")
    existing_lom_dirs = [d for d in lom_data_dirs if os.path.isdir(d)]
    if not existing_lom_dirs:
        print("No valid LOM data directories found. Exiting time interval histogram function.")
        return

    # Sort LOMs to ensure consistent order in processing (e.g. data_muon_run_lom16-01, -02, etc.)
    sorted_lom_dirs = sorted(existing_lom_dirs)

    for lom_idx, current_lom_folder_path in enumerate(tqdm(sorted_lom_dirs, desc="Processing LOMs for Histograms")):
        current_lom_folder_path = os.path.normpath(current_lom_folder_path)
        current_lom_name = os.path.basename(current_lom_folder_path)

        print(f"\n--- LOM: {current_lom_name} - Time Interval Histograms ---")

        # Setup figure with 4x4 subplots for 16 PMTs
        fig = plt.figure(figsize=(12, 12))
        gs = GridSpec(4, 4, figure=fig, hspace=0.5, wspace=0.3)

        # Adjust y-position for the main title of the figure
        fig.suptitle(f"Time Interval Distribution for {current_lom_name}", fontsize=16, y=0.96)

        pmt_to_infile_map = {}
        current_lom_pmt_files = glob.glob(os.path.join(current_lom_folder_path, "*.hdf"))
        current_lom_pmt_files.sort()
        for fname in current_lom_pmt_files:
            file_basename = os.path.basename(fname)
            pmt_id_match = re.search(r'\.(\d{2,3})\.hdf$', file_basename)
            if pmt_id_match:
                pmt_id = int(pmt_id_match.group(1))
                if 0 <= pmt_id < num_channels:
                    pmt_to_infile_map[pmt_id] = fname

        # Define the custom display order for PMT IDs
        custom_pmt_order = [
            8, 10, 12, 14,
            0, 2, 4, 6,
            1, 3, 5, 7,
            9, 11, 13, 15
        ]

        # Define common bin edges for log10(dt)
        bins = np.linspace(-6, 1, 100) # 100 bins from 10^-6 s to 10^1 s

        # Common Y-axis limits and ticks for Count plot (0-1000)
        common_y_max_count = 1000
        common_y_ticks_count = np.arange(0, common_y_max_count + 1, 200)

        for pmt_idx_in_order, pmt_id in enumerate(custom_pmt_order):
            # Check if PMT file actually exists for this LOM
            if pmt_id not in pmt_to_infile_map:
                warnings.warn(f"PMT {pmt_id} file not found for {current_lom_name}. Skipping its histogram.")
                continue

            # Determine subplot position based on its index in the custom_pmt_order
            row = pmt_idx_in_order // 4
            col = pmt_idx_in_order % 4
            ax = fig.add_subplot(gs[row, col])

            infile_pmt = pmt_to_infile_map.get(pmt_id)
            if infile_pmt:
                try:
                    data_all_fpga = read_hdffile(infile_pmt)
                    fpga_time = np.array(data_all_fpga["data"]["FPGAtime"][()] if "data" in data_all_fpga and "FPGAtime" in data_all_fpga["data"] else [])

                    if fpga_time.size > 1:
                        dt_s = np.diff(fpga_time) / 1e9
                        dt_s_positive = dt_s[dt_s > 0]

                        if dt_s_positive.size > 0:
                            log10_dt = np.log10(dt_s_positive)

                            # Using histogram with density=False (raw counts)
                            counts, bin_edges = np.histogram(log10_dt, bins=bins, density=False) # density=False for raw counts
                            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                            ax.plot(bin_centers, counts, linestyle='-', label=f'pmt{pmt_id}')

                        else:
                            ax.text(0.5, 0.5, 'No positive dt data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10, color='gray')
                            warnings.warn(f"No positive time differences found for {current_lom_name} PMT {pmt_id}.")
                    else:
                        ax.text(0.5, 0.5, 'Not enough timestamps', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10, color='gray')
                        warnings.warn(f"Not enough timestamps found for {current_lom_name} PMT {pmt_id}.")
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error: {e}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=8, color='red')
                    warnings.warn(f"Error processing {current_lom_name} PMT {pmt_id} for histograms: {e}")
            else:
                ax.text(0.5, 0.5, 'File not found', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10, color='gray')
                warnings.warn(f"HDF file not found for {current_lom_name} PMT {pmt_id}.")

            ax.set_title(f"pmt{pmt_id}", fontsize=10, loc='left') # Title aligned left
            ax.set_xlabel("log10(dt[sec])", fontsize=8)
            ax.set_ylabel("Count", fontsize=8) # Set Y-label to "Count"
            ax.tick_params(axis='x', labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.set_xticks(np.arange(-6, 2, 1)) # Set x-ticks from -6 to 1 with step 1
            ax.set_xlim(-6.5, 1.5) # Set x-axis limit
            ax.set_ylim(0, common_y_max_count) # Apply common Y-limit for Count
            ax.set_yticks(common_y_ticks_count) # Apply common Y-ticks for Count

        # Adjust overall plot layout, leaving space at the top for suptitle
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        histogram_filename = os.path.join(output_dir_base, f"{current_lom_name}_Time_Interval_Histograms.pdf")
        fig.savefig(histogram_filename)
        plt.close(fig)
        print(f'Saved Time Interval Histograms for {current_lom_name} to: {histogram_filename}')


# --- New Function for calculating and saving individual PMT Trigger Rates ---
def calculate_and_save_pmt_trigger_rates(lom_data_dirs, num_channels, output_dir_base, output_filename="PMT_Individual_Trigger_Rates.xlsx"):
    print(f"\n--- Calculating and Saving Individual PMT Trigger Rates to {output_filename} ---")

    all_pmt_trigger_data = []

    existing_lom_dirs = [d for d in lom_data_dirs if os.path.isdir(d)]
    if not existing_lom_dirs:
        print("No valid LOM data directories found. Exiting trigger rate calculation.")
        return

    for lom_idx, current_lom_folder_path in enumerate(tqdm(sorted(existing_lom_dirs), desc="Calculating PMT Trigger Rates")):
        current_lom_folder_path = os.path.normpath(current_lom_folder_path)
        current_lom_name = os.path.basename(current_lom_folder_path)

        pmt_to_infile_map = {}
        current_lom_pmt_files = glob.glob(os.path.join(current_lom_folder_path, "*.hdf"))
        current_lom_pmt_files.sort()
        for fname in current_lom_pmt_files:
            file_basename = os.path.basename(fname)
            pmt_id_match = re.search(r'\.(\d{2,3})\.hdf$', file_basename)
            if pmt_id_match:
                pmt_id = int(pmt_id_match.group(1))
                if 0 <= pmt_id < num_channels:
                    pmt_to_infile_map[pmt_id] = fname

        sorted_pmt_ids = sorted(pmt_to_infile_map.keys())

        for pmt_id in sorted_pmt_ids:
            infile_pmt = pmt_to_infile_map.get(pmt_id)

            total_events = 0
            livetime_s = 0.0
            trigger_rate = np.nan # Default to NaN

            if infile_pmt:
                try:
                    data_all_fpga = read_hdffile(infile_pmt)
                    fpga_time = np.array(data_all_fpga["data"]["FPGAtime"][()] if "data" in data_all_fpga and "FPGAtime" in data_all_fpga["data"] else [])

                    total_events = fpga_time.size
                    if total_events > 1:
                        start_t = fpga_time[0]
                        end_t = fpga_time[-1]
                        livetime_s = (end_t - start_t) / 1e9 # Convert ns to seconds
                        if livetime_s <= 0: livetime_s = 0.0 # Ensure non-negative livetime

                    if livetime_s > 0:
                        trigger_rate = total_events / livetime_s
                    elif total_events == 0:
                        trigger_rate = 0.0 # If no events, rate is 0

                    # --- DEBUG PRINT for individual PMT rates ---
                    print(f"      LOM {current_lom_name}, PMT {pmt_id}: {total_events} events / {livetime_s:.2f} s = {trigger_rate:.4f} Hz")

                except Exception as e:
                    warnings.warn(f"Error processing trigger rate for {current_lom_name} PMT {pmt_id}: {e}")
            else:
                warnings.warn(f"HDF file not found for {current_lom_name} PMT {pmt_id} for trigger rate calculation.")

            all_pmt_trigger_data.append({
                "LOM": current_lom_name,
                "PMT_ID": pmt_id,
                "Total_Events": total_events,
                "Livetime_s": livetime_s,
                "Trigger_Rate_Hz": trigger_rate
            })

    if not all_pmt_trigger_data:
        print("No individual PMT trigger data collected. Exiting.")
        return

    df_pmt_trigger_rates = pd.DataFrame(all_pmt_trigger_data)

    # Save to XLSX
    xlsx_path = os.path.join(output_dir_base, output_filename)
    try:
        df_pmt_trigger_rates.to_excel(xlsx_path, index=False)
        print(f"Individual PMT Trigger Rates saved to: {xlsx_path}")
    except ImportError:
        warnings.warn("'openpyxl' is not installed. Cannot save to .xlsx. Please install with: pip install openpyxl")
    except Exception as e:
        warnings.warn(f"Error saving individual PMT trigger rates to .xlsx file {xlsx_path}: {e}")

    return df_pmt_trigger_rates # Optionally return the DataFrame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Coincidence Rate vs. Threshold plots for selected PMT pairs.")
    parser.add_argument("lom_data_dirs", nargs='+', help="Input LOM data directories (e.g., 'path/to/LOM16-01/'). Each directory should contain HDF files for PMT channels and the equations summary TXT file.")
    parser.add_argument("--output_dir", "-o", help="Output directory for plots (default: ./plots_output)", type=str, default='./plots_output')
    parser.add_argument("--num_channels", type=int, default=16, help="Total number of PMT channels (e.g., 0-15).")
    args = parser.parse_args()

    # --- Configuration Parameters ---
    FIXED_COINCIDENCE_WINDOW_NS = 100.0
    COINCIDENCE_THRESHOLDS_PE = np.arange(0, 201, 25)

    # === Data for Geant4 Simulation (provided by user) ===
    # This data will be plotted for the "Geant4 simulation" line
    # Provided by user
    geant4_x = np.array([25, 50, 75, 100, 125, 150])
    geant4_y = np.array([0.17299703545767792, 0.08063421144213802, 0.06597344572538565, 0.060109139438684706, 0.04984660343695805, 0.007330382858376183])
    geant4_yerr = np.array([0.09221143808358809, 0.043317539042922425, 0.03570129752279186, 0.03301114842906276, 0.028503786315113924, 0.006044788560260421])

    # Pack Geant4 data into a dictionary to pass to the function
    geant4_simulation_data_for_plot = {
        'x': geant4_x,
        'y': geant4_y,
        'yerr': geant4_yerr
    }
    # ====================================================

    # === Dead Time for Correction (in microseconds) ===
    DEAD_TIME_FOR_PLOT = DEAD_TIME_MU_S


    # === ชุด PMT คู่เดิม (14-15, 12-13, 10-11, 8-9) ===
    SELECTED_PMT_PAIRS_SET1_WITH_MARKERS = {
        (14, 15): 'o',
        (12, 13): '+',
        (10, 11): 'x',
        (8, 9): '1'
    }

    # === ชุด PMT คู่ใหม่ (6-7, 4-5, 2-3, 0-1) เรียงจากมากไปน้อย ===
    SELECTED_PMT_PAIRS_SET2_WITH_MARKERS = {
        (6, 7): '*',
        (4, 5): '^',
        (2, 3): 's',
        (0, 1): 'o'
    }

    output_base_dir = args.output_dir
    os.makedirs(output_base_dir, exist_ok=True)

    print(f"Output plots will be saved to: {output_base_dir}")

    # --- Generate Plot for Original PMT Pairs ---
    print("\n--- Generating Plot for Original PMT Pairs ---")
    plot_coincidence_rate_vs_threshold(
        args.lom_data_dirs,
        output_base_dir,
        FIXED_COINCIDENCE_WINDOW_NS,
        COINCIDENCE_THRESHOLDS_PE,
        num_channels=args.num_channels,
        selected_pmt_pairs_with_markers=SELECTED_PMT_PAIRS_SET1_WITH_MARKERS,
        dead_time_mu_s=DEAD_TIME_FOR_PLOT,
        output_filename_suffix="_Selected_PMT_Pairs",
        geant4_simulation_data=geant4_simulation_data_for_plot
    )

    # --- Generate Plot for Additional PMT Pairs ---
    print("\n--- Generating Plot for Additional PMT Pairs ---")
    plot_coincidence_rate_vs_threshold(
        args.lom_data_dirs,
        output_base_dir,
        FIXED_COINCIDENCE_WINDOW_NS,
        COINCIDENCE_THRESHOLDS_PE,
        num_channels=args.num_channels,
        selected_pmt_pairs_with_markers=SELECTED_PMT_PAIRS_SET2_WITH_MARKERS,
        dead_time_mu_s=DEAD_TIME_FOR_PLOT,
        output_filename_suffix="_Additional_PMT_Pairs",
        geant4_simulation_data=geant4_simulation_data_for_plot
    )

    # --- Generate Time Interval Histograms ---
    plot_time_interval_histograms(
        args.lom_data_dirs,
        args.num_channels,
        output_base_dir
    )

    # --- New: Calculate and Save Individual PMT Trigger Rates ---
    calculate_and_save_pmt_trigger_rates(
        args.lom_data_dirs,
        args.num_channels,
        output_base_dir,
        output_filename="PMT_Individual_Trigger_Rates.xlsx"
    )

    print("\n========================================================")
    print("=== All Analysis and Plots Complete.                  ===")
    print("========================================================")