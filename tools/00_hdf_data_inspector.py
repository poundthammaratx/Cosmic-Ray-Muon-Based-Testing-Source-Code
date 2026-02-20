# hdf_data_inspector.py
#
# A script to inspect the structure and content of HDF data files
# (specifically for SPE and Muon data from PMTs).
# It reads an HDF file, prints its internal structure (keys),
# provides basic statistics of key data arrays, and plots
# example raw waveforms for visual inspection.
#
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore") # Suppress warnings, e.g., from h5py

# === START: Path and Import Management ===
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the 'util' subdirectory
util_path = os.path.join(script_dir, "util")
# Add the 'util' directory to sys.path so Python can find modules inside it
if util_path not in sys.path:
    sys.path.append(util_path)

# Import necessary modules from the 'util' directory
# Assuming HDFWriterModuleInspection contains a function to read HDF files into a dictionary.
from HDFWriterModuleInspection import load_dict

# Ensure read_hdffile is defined, if not, assume load_dict serves this purpose.
# This block makes 'read_hdffile' an alias to 'load_dict',
# mimicking how it's used in the main analysis script.
try:
    # This line might exist in other imported files, so we check first.
    read_hdffile 
except NameError:
    read_hdffile = load_dict # Make read_hdffile an alias for load_dict
# === END: Path and Import Management ===


def inspect_hdf_file(filepath, num_examples=3):
    """
    Reads an HDF file, prints its structure, basic data statistics,
    and plots example raw waveforms.

    Args:
        filepath (str): Path to the HDF file to inspect.
        num_examples (int): Number of example waveforms to plot.
    """
    if not os.path.isfile(filepath):
        print(f"Error: File not found at '{filepath}'. Please check the path.")
        return

    print(f"\n=== Inspecting HDF File: {os.path.basename(filepath)} ===")
    print(f"Full Path: {filepath}")

    try:
        # Load the HDF file data into a dictionary using the read_hdffile alias.
        # CRITICAL FIX: Removed 'quiet=True' from this specific script's call,
        # as the 'load_dict' in user's HDFWriterModuleInspection.py does not support it.
        # The main analysis script's 'read_hdffile' (from eventHist.py) might have a different signature.
        data_dict = read_hdffile(filepath) 

        # --- 1. Display Top-Level Keys ---
        print("\n--- Top-Level Keys in HDF File ---")
        if isinstance(data_dict, dict):
            for key in data_dict.keys():
                print(f"- /{key}")
        else:
            print("Could not load HDF file into expected dictionary format.")
            return

        # --- 2. Inspect 'metadata' Group ---
        print("\n--- Metadata Group (/metadata) ---")
        if "metadata" in data_dict and isinstance(data_dict["metadata"], dict):
            for key, value in data_dict["metadata"].items():
                # For h5py, values might be h5py.Dataset, need to access content
                if hasattr(value, 'value'): # For older h5py versions
                    display_value = value.value
                elif hasattr(value, '__array__'): # For current h5py versions Dataset behaves like numpy array
                    display_value = value[()] # Access dataset content
                else:
                    display_value = value # Regular dictionary value
                
                # Truncate long string values for display
                if isinstance(display_value, (bytes, str)) and len(str(display_value)) > 100:
                    display_value = str(display_value)[:100] + "..."
                
                print(f"  - {key}: {display_value} (Type: {type(display_value).__name__})")
        else:
            print("No 'metadata' group found or it's not a dictionary.")

        # --- 3. Inspect 'data' Group and Basic Statistics ---
        print("\n--- Data Group (/data) and Basic Statistics ---")
        if "data" in data_dict and isinstance(data_dict["data"], dict):
            for key, dataset in data_dict["data"].items():
                if hasattr(dataset, 'shape'): # Check if it's an h5py Dataset (like an array)
                    print(f"  - {key}: Shape = {dataset.shape}, Dtype = {dataset.dtype}")
                    # Optionally, print min/max/mean for numerical data
                    if np.issubdtype(dataset.dtype, np.number) and dataset.size > 0:
                        try:
                            # Access the data to compute stats, handle large datasets
                            if dataset.size > 1e7: # For very large datasets, sample
                                stats_data = dataset[:100000] # Take first 100k samples
                            else:
                                stats_data = dataset[()]
                            print(f"    (Min: {np.min(stats_data):.2f}, Max: {np.max(stats_data):.2f}, Mean: {np.mean(stats_data):.2f})")
                        except Exception as e:
                            print(f"    (Could not compute stats: {e})")
                else:
                    print(f"  - {key}: Not an array-like dataset (Type: {type(dataset).__name__})")
            
            # Get number of events (assuming from FPGAtime or charge_ch1)
            num_events = 0
            if "FPGAtime" in data_dict["data"] and hasattr(data_dict["data"]["FPGAtime"], 'shape'):
                num_events = data_dict["data"]["FPGAtime"].shape[0]
            elif "charge_ch1" in data_dict["data"] and hasattr(data_dict["data"]["charge_ch1"], 'shape'):
                num_events = data_dict["data"]["charge_ch1"].shape[0]
            print(f"\nTotal Number of Events (approx): {num_events}")

        else:
            print("No 'data' group found or it's not a dictionary.")
        
        # --- 4. Plot Example Waveforms ---
        print(f"\n--- Plotting {num_examples} Example Waveforms (Ch1 & Ch2) ---")
        if "data" in data_dict and "ADC_ch1" in data_dict["data"] and "ADC_ch2" in data_dict["data"] and "nsample" in data_dict["data"]:
            adc_ch1_data = data_dict["data"]["ADC_ch1"]
            adc_ch2_data = data_dict["data"]["ADC_ch2"]
            nsample_data = data_dict["data"]["nsample"]
            fpga_time_data = data_dict["data"]["FPGAtime"] if "FPGAtime" in data_dict["data"] else None

            # Select a few random event indices for plotting
            if num_events > 0:
                plot_indices = np.random.choice(num_events, min(num_examples, num_events), replace=False)
            else:
                plot_indices = []

            if not plot_indices:
                print("No events to plot examples.")
                return

            fig, axes = plt.subplots(num_examples, 2, figsize=(12, 4 * num_examples), squeeze=False)
            fig.suptitle(f"Example Waveforms from {os.path.basename(filepath)}", fontsize=16)

            for i, ev_idx in enumerate(plot_indices):
                n_samples = nsample_data[ev_idx]
                time_ns = np.array([j * (1e9 / 60e6) for j in range(n_samples)]) # Assuming 60 MSPS
                
                # Channel 1 plot
                ax1 = axes[i, 0]
                ax1.plot(time_ns, adc_ch1_data[ev_idx][:n_samples], color='blue')
                ax1.set_title(f"Event {ev_idx} (Ch1)", fontsize=10)
                ax1.set_xlabel("Time (ns)", fontsize=8)
                ax1.set_ylabel("ADC Counts", fontsize=8)
                ax1.set_ylim([0, 4096]) # Assuming 12-bit ADC
                ax1.grid(True, linestyle=':', alpha=0.7)

                # Channel 2 plot
                ax2 = axes[i, 1]
                ax2.plot(time_ns, adc_ch2_data[ev_idx][:n_samples], color='red')
                ax2.set_title(f"Event {ev_idx} (Ch2)", fontsize=10)
                ax2.set_xlabel("Time (ns)", fontsize=8)
                ax2.set_ylabel("ADC Counts", fontsize=8)
                ax2.set_ylim([0, 4096]) # Assuming 12-bit ADC
                ax2.grid(True, linestyle=':', alpha=0.7)
                
                if fpga_time_data is not None:
                    ax1.text(0.98, 0.98, f'FPGA Time: {fpga_time_data[ev_idx]}', transform=ax1.transAxes,
                             fontsize=7, ha='right', va='top', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))


            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
            output_dir_plots = "hdf_inspection_plots"
            if not os.path.exists(output_dir_plots):
                os.makedirs(output_dir_plots)
            plot_filename = os.path.join(output_dir_plots, f"example_waveforms_{os.path.basename(filepath).replace('.', '_')}.pdf")
            fig.savefig(plot_filename)
            plt.close(fig)
            print(f"Example waveforms saved to: {plot_filename}")

        else:
            print("Necessary waveform data (ADC_ch1, ADC_ch2, nsample) not found in '/data' group.")

    except Exception as e:
        print(f"An error occurred during file inspection: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspects HDF data files (SPE/Muon) to show structure, stats, and example waveforms."
    )
    parser.add_argument(
        "filepath",
        type=str,
        help="Path to the HDF data file (e.g., 'data_muon_run909_00.hdf' or 'spe_sample_data/data-spe-run905-0000.00.hdf')"
    )
    # Corrected default value from 's' to a number (3)
    parser.add_argument(
        "--num_examples",
        type=int,
        default=3, 
        help="Number of example waveforms to plot (default: 3)."
    )
    args = parser.parse_args()

    inspect_hdf_file(args.filepath, args.num_examples)
