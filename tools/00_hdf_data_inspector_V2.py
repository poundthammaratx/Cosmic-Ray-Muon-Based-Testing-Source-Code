# 00_hdf_data_inspector_V2.py
#
# A script to inspect the structure and content of HDF data files
# (specifically for SPE and Muon data from PMTs).
# It reads an HDF file, prints its internal structure (keys),
# provides basic statistics of key data arrays, and plots
# example raw waveforms for visual inspection.
#
# This is a V2 version that uses a dedicated HDF file reader module
# to robustly load the entire HDF5 file content into a dictionary.
# It can process a single HDF file or all HDF files within a specified directory.
#
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import warnings
import glob # Required for searching files in a directory
import pandas as pd # <-- เพิ่ม import pandas

# Suppress warnings, e.g., from h5py (often related to deprecations or internal structures)
warnings.filterwarnings("ignore") 

# === START: Path and Import Management ===
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the 'util' subdirectory
util_path = os.path.join(script_dir, "util")

# Add the 'util' directory to sys.path so Python can find modules inside it.
# This is crucial for importing 'hdf_reader'.
if util_path not in sys.path:
    sys.path.append(util_path)

# Import the new, dedicated HDF file reader function.
# This function is responsible for opening and parsing the entire HDF5 file.
from hdf_reader import load_hdf_file_as_dict

# Define 'read_hdffile' as an alias for our new, correct loading function.
# This maintains compatibility with the rest of the script's logic if it was
# previously designed to use a function named 'read_hdffile'.
read_hdffile = load_hdf_file_as_dict 
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
        # Load the HDF file data into a dictionary using the dedicated function.
        # This is the central point of loading the HDF data.
        data_dict = read_hdffile(filepath) 

        # --- Convert Data to Tabular Format and Save to CSV ---
        print("\n--- Converting Data to Tabular Format ---")
        if "data" in data_dict and isinstance(data_dict["data"], dict):
            data_for_df = {}
            # List of 1D datasets to include in the DataFrame
            one_d_datasets = [
                "nsample", "FPGAtime", "FPGAtcword",
                "charge_ch1", "peak_ch1", "time_ch1", "pedestal_ch1",
                "charge_fit_ch1", "peak_fit_ch1", "time_fit_ch1",
                "charge_ch2", "peak_ch2", "time_ch2", "pedestal_ch2",
                "charge_fit_ch2", "peak_fit_ch2", "time_fit_ch2"
            ]

            for key in one_d_datasets:
                if key in data_dict["data"] and isinstance(data_dict["data"][key], np.ndarray) and data_dict["data"][key].ndim == 1:
                    data_for_df[key] = data_dict["data"][key]
                elif key in data_dict["data"] and isinstance(data_dict["data"][key], np.ndarray) and data_dict["data"][key].ndim > 1:
                    print(f"Skipping 2D or higher dimensional dataset: '{key}' for direct tabular conversion to avoid excessively wide tables.")
                else:
                    print(f"Warning: Dataset '{key}' not found or not in expected 1D array format. It will not be included in the DataFrame.")

            if not data_for_df:
                print("No suitable 1D datasets found to create a DataFrame from the '/data' group.")
            else:
                try:
                    df = pd.DataFrame(data_for_df)

                    # Define output directory for CSVs
                    csv_output_dir = "hdf_tabular_data"
                    if not os.path.exists(csv_output_dir):
                        os.makedirs(csv_output_dir)

                    # Create a unique filename for the CSV based on the HDF file
                    csv_filename = os.path.join(csv_output_dir, f"{os.path.basename(filepath).replace('.hdf', '')}_summary.csv")

                    df.to_csv(csv_filename, index=False)
                    print(f"Tabular data for '{os.path.basename(filepath)}' saved to: {csv_filename}")
                    print("\nFirst 5 rows of the generated DataFrame:")
                    print(df.head())
                except Exception as e:
                    print(f"Error creating or saving DataFrame for '{filepath}': {e}")
        else:
            print("No 'data' group found in the HDF file to create tabular data.")
        # --- End of Tabular Conversion Section ---

        # --- 1. Display Top-Level Keys ---
        print("\n--- Top-Level Keys in HDF File ---")
        if isinstance(data_dict, dict) and data_dict: # Check if it's a non-empty dictionary
            for key in data_dict.keys():
                print(f"- /{key}")
        else:
            print("Could not load HDF file into expected dictionary format or file is empty.")
            return # Exit if data_dict is not valid

        # --- 2. Inspect 'metadata' Group ---
        print("\n--- Metadata Group (/metadata) ---")
        if "metadata" in data_dict and isinstance(data_dict["metadata"], dict):
            for key, value in data_dict["metadata"].items():
                display_value = value

                # Truncate long string values for display for readability
                if isinstance(display_value, str) and len(display_value) > 100:
                    display_value = display_value[:100] + "..."
                elif isinstance(display_value, bytes) and len(display_value) > 100:
                    # Attempt to decode bytes for display if still in bytes format
                    try:
                        display_value = display_value.decode('utf-8')[:100] + "..."
                    except (UnicodeDecodeError, AttributeError):
                        pass # Keep as bytes representation if decoding fails

                print(f"   - {key}: {display_value} (Type: {type(display_value).__name__})")
        else:
            print("No 'metadata' group found or it's not a dictionary.")

        # --- 3. Inspect 'data' Group and Basic Statistics ---
        print("\n--- Data Group (/data) and Basic Statistics ---")
        if "data" in data_dict and isinstance(data_dict["data"], dict):
            for key, dataset_array in data_dict["data"].items():
                # Check if it's a numpy array or similar array-like object
                if hasattr(dataset_array, 'shape') and hasattr(dataset_array, 'dtype'):
                    print(f"   - {key}: Shape = {dataset_array.shape}, Dtype = {dataset_array.dtype}")
                    # Optionally, print min/max/mean for numerical data
                    if np.issubdtype(dataset_array.dtype, np.number) and dataset_array.size > 0:
                        try:
                            # For very large datasets, sample a portion for statistics
                            if dataset_array.size > 1e7: # Example threshold: 10 million elements
                                stats_data = dataset_array[:100000] # Take first 100k samples for stats
                            else:
                                stats_data = dataset_array # Use full data for smaller arrays
                            print(f"     (Min: {np.min(stats_data):.2f}, Max: {np.max(stats_data):.2f}, Mean: {np.mean(stats_data):.2f})")
                        except Exception as e:
                            print(f"     (Could not compute stats: {e}) - {e}")
                else:
                    print(f"   - {key}: Not an array-like dataset (Type: {type(dataset_array).__name__})")

            # Get total number of events, typically derived from a common data array like FPGAtime
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
        # Check for essential waveform data existence
        if ("data" in data_dict and 
            "ADC_ch1" in data_dict["data"] and 
            "ADC_ch2" in data_dict["data"] and 
            "nsample" in data_dict["data"]):

            adc_ch1_data = data_dict["data"]["ADC_ch1"]
            adc_ch2_data = data_dict["data"]["ADC_ch2"]
            nsample_data = data_dict["data"]["nsample"]
            fpga_time_data = data_dict["data"]["FPGAtime"] if "FPGAtime" in data_dict["data"] else None

            # Ensure data arrays are not empty
            if num_events == 0 or adc_ch1_data.size == 0 or adc_ch2_data.size == 0 or nsample_data.size == 0:
                print("No waveform data available for plotting.")
                return

            plot_indices = np.random.choice(num_events, min(num_examples, num_events), replace=False)

            if not plot_indices.tolist(): # Check if the list of indices is empty
                print("No events selected for plotting examples.")
                return

            fig, axes = plt.subplots(num_examples, 2, figsize=(12, 4 * num_examples), squeeze=False)
            fig.suptitle(f"Example Waveforms from {os.path.basename(filepath)}", fontsize=16)

            for i, ev_idx in enumerate(plot_indices):
                n_samples = nsample_data[ev_idx]
                # Assuming 60 MSPS (Mega Samples Per Second) for time calculation
                # 1 sample time = 1 / 60e6 seconds = (1 / 60e6) * 1e9 ns
                time_ns = np.array([j * (1e9 / 60e6) for j in range(n_samples)]) 

                # Channel 1 plot
                ax1 = axes[i, 0]
                ax1.plot(time_ns, adc_ch1_data[ev_idx][:n_samples], color='blue')
                ax1.set_title(f"Event {ev_idx} (Ch1)", fontsize=10)
                ax1.set_xlabel("Time (ns)", fontsize=8)
                ax1.set_ylabel("ADC Counts", fontsize=8)
                ax1.set_ylim([0, 4096]) # Assuming 12-bit ADC (0 to 2^12-1)
                ax1.grid(True, linestyle=':', alpha=0.7)

                # Channel 2 plot
                ax2 = axes[i, 1]
                ax2.plot(time_ns, adc_ch2_data[ev_idx][:n_samples], color='red')
                ax2.set_title(f"Event {ev_idx} (Ch2)", fontsize=10)
                ax2.set_xlabel("Time (ns)", fontsize=8)
                ax2.set_ylabel("ADC Counts", fontsize=8)
                ax2.set_ylim([0, 4096]) # Assuming 12-bit ADC
                ax2.grid(True, linestyle=':', alpha=0.7)

                # Display FPGA Time if available
                if fpga_time_data is not None and ev_idx < fpga_time_data.size:
                    ax1.text(0.98, 0.98, f'FPGA Time: {fpga_time_data[ev_idx]}', transform=ax1.transAxes,
                             fontsize=7, ha='right', va='top', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

            # Adjust layout to prevent titles/labels from overlapping
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle

            # Save the plots to a dedicated directory
            output_dir_plots = "hdf_inspection_plots"
            if not os.path.exists(output_dir_plots):
                os.makedirs(output_dir_plots)

            # Create a unique filename for the plot PDF
            plot_filename = os.path.join(output_dir_plots, f"example_waveforms_{os.path.basename(filepath).replace('.', '_')}.pdf")
            fig.savefig(plot_filename)
            plt.close(fig) # Close the figure to free up memory
            print(f"Example waveforms saved to: {plot_filename}")

        else:
            print("Necessary waveform data (ADC_ch1, ADC_ch2, nsample) not found or is empty in '/data' group. Cannot plot.")

    except Exception as e:
        print(f"An unexpected error occurred during file inspection for '{filepath}': {e}")
        # import traceback; traceback.print_exc() # Uncomment for more detailed debug info


if __name__ == "__main__":
    # --- Program execution starts here ---

    # Set up argument parser for command-line arguments
    parser = argparse.ArgumentParser(
        description="Inspects HDF data files (SPE/Muon) to show structure, stats, and example waveforms."
    )
    # Define the required 'path' argument, which can be a single file or a directory
    parser.add_argument(
        "path",
        type=str,
        help="Path to the HDF data file (e.g., 'data_muon_run909_00.hdf') OR a directory containing HDF files (e.g., 'spe_sample_data/')"
    )
    # Define the optional '--num_examples' argument with a default value
    parser.add_argument(
        "--num_examples",
        type=int,
        default=3, 
        help="Number of example waveforms to plot for each file (default: 3)."
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Determine if the provided path is a single file or a directory
    if os.path.isfile(args.path):
        # If it's a single file, proceed with inspection
        inspect_hdf_file(args.path, args.num_examples)
    elif os.path.isdir(args.path):
        # If it's a directory, search for .hdf files within it
        print(f"Searching for .hdf files in directory: {args.path}")
        # Use glob to find all files ending with .hdf in the specified directory
        hdf_files = glob.glob(os.path.join(args.path, "*.hdf"))
        hdf_files.sort() # Sort the list of files for consistent processing order

        if not hdf_files:
            # Inform if no HDF files are found
            print(f"No .hdf files found in the directory: {args.path}")
        else:
            # Process each found HDF file
            print(f"Found {len(hdf_files)} .hdf files. Processing...")
            for hdf_file in hdf_files:
                inspect_hdf_file(hdf_file, args.num_examples)
    else:
        # Handle cases where the path is neither a file nor a directory
        print(f"Error: Provided path '{args.path}' is neither a file nor a directory. Please provide a valid file or directory path.")
        sys.exit(1)