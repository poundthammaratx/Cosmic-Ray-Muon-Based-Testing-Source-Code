# pmt_data_characterization.py
#
# Script for analyzing the characteristics of PMT data from summarized CSV files,
# specifically for PMT 08 from SPE and Muon data.
# It will read CSV files, display basic DataFrame statistics,
# and plot charge distribution graphs and channel correlation graphs.
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import warnings
warnings.filterwarnings("ignore") # Suppress unnecessary warnings

def characterize_pmt_data(spe_filepath, muon_filepath, output_dir='./pmt_characterization_plots'):
    """
    Reads CSV files for SPE and Muon data for a single PMT,
    displays basic statistics, and generates various characterization plots.

    Args:
        spe_filepath (str): Path to the SPE data CSV file (e.g., data-spe-run905-0000.08_summary.csv).
        muon_filepath (str): Path to the Muon data CSV file (e.g., data_muon_run909_08_summary.csv).
        output_dir (str): Directory for saving plot results.
    """

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"=== Starting PMT 08 Data Characterization ===")
    print(f"SPE File: {spe_filepath}")
    print(f"Muon File: {muon_filepath}")
    print(f"Output Directory: {output_dir}")

    # --- 1. Load Data from CSV Files ---
    try:
        # โหลด SPE Data
        df_spe = pd.read_csv(spe_filepath)
        print(f"\n--- Successfully loaded SPE Data for PMT 08. Number of events: {len(df_spe)} ---")
        print("First 5 rows of SPE Data:")
        print(df_spe.head())

        # โหลด Muon Data
        df_muon = pd.read_csv(muon_filepath)
        print(f"\n--- Successfully loaded Muon Data for PMT 08. Number of events: {len(df_muon)} ---")
        print("First 5 rows of Muon Data:")
        print(df_muon.head())

    except FileNotFoundError as e:
        print(f"Error: File not found. Please check the file path: {e}")
        return
    except Exception as e:
        print(f"Error: An error occurred while loading CSV files: {e}")
        return

    # --- 2. Display Basic DataFrame Statistics ---
    print("\n--- Descriptive Statistics for PMT 08 ---")
    # Corrected column names to match the CSV files
    print("\n[SPE Data - charge_ch1 & charge_ch2]")
    print(df_spe[['charge_ch1', 'charge_ch2']].describe().round(3))

    print("\n[Muon Data - charge_ch1 & charge_ch2]")
    print(df_muon[['charge_ch1', 'charge_ch2']].describe().round(3))

    # --- 3. Charge Distribution Histograms ---
    print("\n--- Generating Charge Distribution Plots... ---")

    # Plot for SPE Data (focus on 1 PE Peak)
    fig_spe_charge, axes_spe_charge = plt.subplots(1, 2, figsize=(14, 6))
    fig_spe_charge.suptitle('PMT 08 SPE Charge Distributions (Log Y-Scale)', fontsize=16)

    # Ch1 SPE - Corrected column name
    axes_spe_charge[0].hist(df_spe['charge_ch1'], bins=100, range=[-0.5, 5.0], color='skyblue', edgecolor='gray')
    axes_spe_charge[0].set_title('Channel 1', fontsize=12)
    axes_spe_charge[0].set_xlabel('Charge (pC)', fontsize=10)
    axes_spe_charge[0].set_ylabel('Counts (Log Scale)', fontsize=10)
    axes_spe_charge[0].set_yscale('log')
    axes_spe_charge[0].grid(True, linestyle=':', alpha=0.6)
    # Add an expected 1 PE line (approx 0.801 pC)
    axes_spe_charge[0].axvline(x=0.801, color='red', linestyle='--', label='~1 PE')
    axes_spe_charge[0].legend()


    # Ch2 SPE - Corrected column name
    axes_spe_charge[1].hist(df_spe['charge_ch2'], bins=100, range=[-0.5, 5.0], color='lightcoral', edgecolor='gray')
    axes_spe_charge[1].set_title('Channel 2', fontsize=12)
    axes_spe_charge[1].set_xlabel('Charge (pC)', fontsize=10)
    axes_spe_charge[1].set_ylabel('Counts (Log Scale)', fontsize=10)
    axes_spe_charge[1].set_yscale('log')
    axes_spe_charge[1].grid(True, linestyle=':', alpha=0.6)
    axes_spe_charge[1].axvline(x=0.801, color='red', linestyle='--', label='~1 PE')
    axes_spe_charge[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'PMT08_SPE_Charge_Distributions.pdf'))
    plt.close(fig_spe_charge)
    print(f"Saved SPE Charge Distributions plot to: {os.path.join(output_dir, 'PMT08_SPE_Charge_Distributions.pdf')}")


    # Plot for Muon Data (focus on 100 pC Peak and wide range)
    fig_muon_charge, axes_muon_charge = plt.subplots(1, 2, figsize=(14, 6))
    fig_muon_charge.suptitle('PMT 08 Muon Charge Distributions (Log Y-Scale)', fontsize=16)

    # Ch1 Muon - Corrected column name
    axes_muon_charge[0].hist(df_muon['charge_ch1'], bins=100, range=[-10, 300], color='lightgreen', edgecolor='gray')
    axes_muon_charge[0].set_title('Channel 1', fontsize=12)
    axes_muon_charge[0].set_xlabel('Charge (pC)', fontsize=10)
    axes_muon_charge[0].set_ylabel('Counts (Log Scale)', fontsize=10)
    axes_muon_charge[0].set_yscale('log')
    axes_muon_charge[0].grid(True, linestyle=':', alpha=0.6)
    # Add an expected 100 pC peak line
    axes_muon_charge[0].axvline(x=100, color='red', linestyle='--', label='100 pC Peak')
    axes_muon_charge[0].legend()


    # Ch2 Muon - Corrected column name
    axes_muon_charge[1].hist(df_muon['charge_ch2'], bins=100, range=[-10, 300], color='lightsalmon', edgecolor='gray')
    axes_muon_charge[1].set_title('Channel 2', fontsize=12)
    axes_muon_charge[1].set_xlabel('Charge (pC)', fontsize=10)
    axes_muon_charge[1].set_ylabel('Counts (Log Scale)', fontsize=10)
    axes_muon_charge[1].set_yscale('log')
    axes_muon_charge[1].grid(True, linestyle=':', alpha=0.6)
    axes_muon_charge[1].axvline(x=100, color='red', linestyle='--', label='100 pC Peak')
    axes_muon_charge[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'PMT08_Muon_Charge_Distributions.pdf'))
    plt.close(fig_muon_charge)
    print(f"Saved Muon Charge Distributions plot to: {os.path.join(output_dir, 'PMT08_Muon_Charge_Distributions.pdf')}")


    # --- 4. Channel Correlation Plot (Charge Correlation - Ch1 vs Ch2) ---
    print("\n--- Generating Channel 1 and 2 Charge Correlation Plot... ---")

    fig_correlation, ax_correlation = plt.subplots(1, 1, figsize=(8, 8))
    fig_correlation.suptitle('PMT 08 Charge Correlation (Ch1 vs Ch2) - Muon Data', fontsize=14)

    # Ch1 vs Ch2 Correlation - Corrected column names
    ax_correlation.scatter(df_muon['charge_ch1'], df_muon['charge_ch2'], 
                           alpha=0.5, s=5, color='darkblue') # alpha for scatter density, s for marker size
    ax_correlation.set_xlabel('Charge Channel 1 (pC)', fontsize=10)
    ax_correlation.set_ylabel('Charge Channel 2 (pC)', fontsize=10)
    ax_correlation.set_title(f"Correlation Coefficient: {df_muon['charge_ch1'].corr(df_muon['charge_ch2']):.3f}", fontsize=12) # Corrected column names for correlation
    ax_correlation.grid(True, linestyle=':', alpha=0.6)
    ax_correlation.set_xlim([-10, 300]) # Consistent with Muon charge range
    ax_correlation.set_ylim([-10, 300])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'PMT08_Charge_Correlation.pdf'))
    plt.close(fig_correlation)
    print(f"Saved Charge Correlation plot to: {os.path.join(output_dir, 'PMT08_Charge_Correlation.pdf')}")

    print("\n=== PMT 08 Data Characterization Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Characterize PMT data from SPE and Muon CSV summary files."
    )
    parser.add_argument(
        "spe_filepath",
        type=str,
        help="Path to the SPE data CSV file (e.g., 'data-spe-run905-0000.08_summary.csv')"
    )
    parser.add_argument(
        "muon_filepath",
        type=str,
        help="Path to the Muon data CSV file (e.g., 'data_muon_run909_08_summary.csv')"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./pmt_characterization_plots',
        help="Output directory for plots (default: './pmt_characterization_plots')"
    )
    args = parser.parse_args()

    # Call the analysis function
    characterize_pmt_data(args.spe_filepath, args.muon_filepath, args.output_dir)
