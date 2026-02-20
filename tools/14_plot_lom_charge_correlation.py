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
import scipy.fft
import re
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm # Import colormap for scatter plot

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

# Define 'read_hdffile' as an alias for our HDF loading function.
read_hdffile = load_hdf_file_as_dict
# === END: HDF File Loading Function ===


# --- START: GLOBAL CONFIGURATION FOR SECTIONS ---
# Sections for the main overview plot (Ch1 on X-axis)
CH1_OVERVIEW_SECTIONS = [
    (20.0, 30.0),
    (31.0, 40.0),
    (41.0, 50.0),
    (51.0, 60.0),
    (61.0, 67.2)
]

# Sections for the NEW aggregated error bar plots (Ch2 on X-axis)
CH2_AGGREGATION_SECTIONS = [
    (0.0, 1.0),
    (1.0, 2.0),
    (2.0, 3.0),
    (3.0, 4.0),
    (4.0, 5.0),
    (5.0, 6.0),
    (6.0, 7.0),
    (7.0, 8.0),
    (8.0, 9.0),
    (9.0, 10.0)
]
# --- END: GLOBAL CONFIGURATION ---


# --- Start: Helper Functions ---

def linear_function(x, m, c):
    return m * x + c

def polynomial_function(x, a, b, c):
    return a * x**2 + b * x + c

def get_charges_of_these_events(filename, evidx_list):
    """
    Retrieves charge values (in pC) for specific event indices from an HDF file.
    """
    ret_charges_ch1 = []
    ret_charges_ch2 = []

    data = read_hdffile(filename)

    if not data or "data" not in data or "metadata" not in data:
        warnings.warn(f"Missing 'data' or 'metadata' group in {filename}. Skipping timestamp retrieval.")
        return [], []

    # Safely get data with .get()
    q_ch1_raw = data.get("data", {}).get("charge_ch1")
    conversion_ch1 = data.get("metadata", {}).get("conversion_ch1", 1.0)
    q_ch2_raw = data.get("data", {}).get("charge_ch2")
    conversion_ch2 = data.get("metadata", {}).get("conversion_ch2", 1.0)
    fpga_time = data.get("data", {}).get("FPGAtime")

    # Check if any required data is missing or None
    if q_ch1_raw is None or q_ch2_raw is None or fpga_time is None:
        warnings.warn(f"One or more required datasets (charge_ch1, charge_ch2, FPGAtime) missing in '{filename}'. Skipping.")
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

    q_ch1_pC_all = q_ch1_raw * (conversion_ch1 * conversion_factor_pC)
    q_ch2_pC_all = q_ch2_raw * (conversion_ch2 * conversion_factor_pC)

    min_charge_len = min(len(q_ch1_pC_all), len(q_ch2_pC_all), len(fpga_time))

    for iev in evidx_list:
        if iev < min_charge_len:
            ret_charges_ch1.append(q_ch1_pC_all[iev])
            ret_charges_ch2.append(q_ch2_pC_all[iev])
        else:
            ret_charges_ch1.append(np.nan)
            ret_charges_ch2.append(np.nan)
            warnings.warn(f"Event index {iev} out of bounds for charge data (length {min_charge_len}) in {filename}. Appending NaN.")

    return ret_charges_ch1, ret_charges_ch2
# --- End: Helper Functions ---


# --- Start: Plotting Functions ---

# MODIFIED: Function to plot Ch1 vs Ch2 scatter, colored by Ch1 value with section lines
# This function now generates the scatter plot instead of histograms.
def plot_ch1_ch2_scatter_by_ch1_color(df_pmt_events, pmt_full_name, output_dir):
    """
    Generates a scatter plot of Charge Ch1 vs Charge Ch2 for a single PMT,
    where points are colored according to their Ch1 value, and vertical lines
    indicate CH1_OVERVIEW_SECTIONS.
    """
    if df_pmt_events.empty or 'ch1_pC' not in df_pmt_events.columns or 'ch2_pC' not in df_pmt_events.columns:
        warnings.warn(f"Cannot plot Ch1-Ch2 scatter for {pmt_full_name}: DataFrame is empty or missing required columns.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    x_data = df_pmt_events['ch1_pC']
    y_data = df_pmt_events['ch2_pC']

    # Use a colormap to color points by their Ch1 value
    scatter = ax.scatter(x_data, y_data, c=x_data, cmap='viridis', s=5, alpha=0.7,
                         vmin=min(s[0] for s in CH1_OVERVIEW_SECTIONS) if CH1_OVERVIEW_SECTIONS else x_data.min(),
                         vmax=max(s[1] for s in CH1_OVERVIEW_SECTIONS) if CH1_OVERVIEW_SECTIONS else x_data.max()) # Set vmin/vmax based on sections or data range

    # Add vertical lines for CH1_OVERVIEW_SECTIONS boundaries
    if CH1_OVERVIEW_SECTIONS:
        for i, (min_ch1, max_ch1) in enumerate(CH1_OVERVIEW_SECTIONS):
            # Plot min boundary line
            ax.axvline(min_ch1, color='red', linestyle='--', linewidth=1.0, alpha=0.7, 
                       label=f'Ch1 Section Boundaries' if i==0 else "")
            # Plot max boundary line only if it's not the same as the next min boundary
            # This avoids double-plotting lines for contiguous sections
            if i < len(CH1_OVERVIEW_SECTIONS) - 1 and max_ch1 == CH1_OVERVIEW_SECTIONS[i+1][0]:
                pass # The next section's start will cover this line
            else:
                ax.axvline(max_ch1, color='red', linestyle='--', linewidth=1.0, alpha=0.7)

            # Add labels for the sections on the plot itself
            if max_ch1 > min_ch1: # Ensure valid range for text placement
                ax.text((min_ch1 + max_ch1) / 2, ax.get_ylim()[1] * 0.95, f'Sec {i+1}', 
                        horizontalalignment='center', verticalalignment='top', 
                        color='darkred', fontsize=8, alpha=0.8, rotation=90)
    else:
        warnings.warn("CH1_OVERVIEW_SECTIONS is empty, no section boundaries will be drawn on the scatter plot.")


    # Set title and labels
    ax.set_title(f'{pmt_full_name} Charge Correlation (Ch1 vs Ch2) by Ch1 Section', fontsize=14)
    ax.set_xlabel('Charge Ch1 (pC)', fontsize=12)
    ax.set_ylabel('Charge Ch2 (pC)', fontsize=12)
    
    # Set x-lim and y-lim consistent with other correlation plots
    ax.set_xlim(0, 100) # Consistent with 16-PMT overview plot
    ax.set_ylim(0, 10)  # Consistent with 16-PMT overview plot

    ax.grid(True, linestyle=':', alpha=0.6)
    ax.tick_params(labelsize=1)

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, label='Charge Ch1 (pC)')
    # Only add legend if there are explicit labels for lines
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=12, loc='upper left')

    # Save the plot
    output_filename = f"{pmt_full_name.replace(' ', '')}_Ch1_Ch2_Scatter_by_Ch1_Section.pdf"
    plt.savefig(os.path.join(output_dir, output_filename))
    plt.close(fig)
    print(f'Saved Ch1-Ch2 scatter plot by Ch1 section for {pmt_full_name} to: {os.path.join(output_dir, output_filename)}')


def plot_single_section_data_for_aggregation(df_section_events, pmt_full_name, section_range_tuple):
    """
    Calculates aggregated statistics (mean, std dev of Ch1) for events within a Ch2 range.
    """
    ch2_min_section = section_range_tuple[0]
    ch2_max_section = section_range_tuple[1]
    
    x_center_ch2 = np.nan
    y_mean_ch1 = np.nan
    y_std_ch1 = np.nan

    if not df_section_events.empty and 'ch1_pC' in df_section_events.columns and 'ch2_pC' in df_section_events.columns:
        
        x_center_ch2 = (ch2_min_section + ch2_max_section) / 2.0

        ch1_data_in_section = df_section_events['ch1_pC'].copy().values
        
        ch1_data_in_section = ch1_data_in_section[~np.isnan(ch1_data_in_section)]

        if len(ch1_data_in_section) > 0:
            y_mean_ch1 = np.mean(ch1_data_in_section)
            y_std_ch1 = np.std(ch1_data_in_section)
        else:
            warnings.warn(f"No valid numerical Ch1 data in {pmt_full_name} Section (Ch2) [{ch2_min_section:.1f}-{ch2_max_section:.1f}] pC for aggregation.")
    else:
        warnings.warn(f"No events found in {pmt_full_name} Section (Ch2) [{ch2_min_section:.1f}-{ch2_max_section:.1f}] pC for aggregation.")
    
    return x_center_ch2, y_mean_ch1, y_std_ch1


def plot_aggregated_errorbar_and_fit(ax, pmt_full_name, aggregated_data):
    """
    Plots aggregated error bars and a polynomial curve fit for a single PMT.
    """
    x_centers_plot = np.array([d['x_center'] for d in aggregated_data if not np.isnan(d['y_mean']) and not np.isnan(d['x_center'])])
    y_means_plot = np.array([d['y_mean'] for d in aggregated_data if not np.isnan(d['y_mean']) and not np.isnan(d['x_center'])])
    y_stds_plot = np.array([d['y_std'] for d in aggregated_data if not np.isnan(d['y_mean']) and not np.isnan(d['x_center'])])

    if len(x_centers_plot) > 0:
        ax.errorbar(x_centers_plot, y_means_plot, yerr=y_stds_plot, fmt='o', capsize=5, color='red', label='Aggregated Data (Mean ± Std Dev)')

        if len(x_centers_plot) >= 3:
            try:
                params, cov = curve_fit(polynomial_function, x_centers_plot, y_means_plot,
                                        sigma=y_stds_plot, absolute_sigma=True,
                                        p0=[0.001, 0.05, 0.5])
                
                x_fit_curve = np.linspace(min(x_centers_plot), max(x_centers_plot), 100)
                y_fit_curve = polynomial_function(x_fit_curve, *params)
                ax.plot(x_fit_curve, y_fit_curve, color='blue', linestyle='--', label=f'Polynomial Fit (Degree {len(params)-1})')
            except RuntimeError:
                warnings.warn(f"Could not fit polynomial for {pmt_full_name} aggregated data (RuntimeError).")
            except ValueError as e:
                warnings.warn(f"Fitting error for {pmt_full_name} aggregated data (ValueError): {e}")
        else:
            warnings.warn(f"Not enough points ({len(x_centers_plot)}) for polynomial fit for {pmt_full_name} aggregated data.")


        ax.set_title(f'{pmt_full_name}', fontsize=12)
        ax.set_xlabel('Charge Ch2 (pC)', fontsize=12)
        ax.set_ylabel('Charge Ch1 (pC)', fontsize=12)
        
        overall_min_ch2_range = min(s[0] for s in CH2_AGGREGATION_SECTIONS)
        overall_max_ch2_range = max(s[1] for s in CH2_AGGREGATION_SECTIONS)
        ax.set_xlim(overall_min_ch2_range - 0.5, overall_max_ch2_range + 0.5)
        
        if len(y_means_plot) > 0:
            min_y_lim = y_means_plot.min() - (y_stds_plot.max() if len(y_stds_plot)>0 else 0) * 1.5
            max_y_lim = y_means_plot.max() + (y_stds_plot.max() if len(y_stds_plot)>0 else 0) * 1.5
            ax.set_ylim(bottom=max(0, min_y_lim), top=max(100, max_y_lim))
        else:
            ax.set_ylim(0, 100)

        ax.tick_params(labelsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(fontsize=12, loc='upper left')
    else:
        ax.text(0.5, 0.5, "No valid aggregated data.", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=8, color='gray')


def plot_multi_pmt_aggregated_curve_fits_overview(pmt_dfs_for_correlation, lom_name, base_output_dir, lom_output_prefix):
    """
    Orchestrates plotting all 16 PMT aggregated error bar plots onto a single PDF page.
    """
    custom_subplot_order = np.array([8, 10, 12, 14, 0, 2, 4, 6, 1, 3, 5, 7, 9, 11, 13, 15])
    pmt_ids_present = sorted(pmt_dfs_for_correlation.keys())
    pmt_ids_to_plot_ordered = [pid for pid in custom_subplot_order if pid in pmt_ids_present]
    if len(pmt_ids_to_plot_ordered) < 16:
        pmt_ids_to_plot_ordered = sorted(pmt_dfs_for_correlation.keys())

    fig_agg_overview, axes_agg_overview = plt.subplots(4, 4, figsize=(20, 16))
    axes_agg_overview = axes_agg_overview.flatten()

    for i, pmt_id in enumerate(pmt_ids_to_plot_ordered):
        ax_current_pmt_agg = axes_agg_overview[i]
        df_pmt_events = pmt_dfs_for_correlation.get(pmt_id)
        pmt_full_name = f"PMT {pmt_id:02d}"

        aggregated_data_for_errorbar_plot = []
        if df_pmt_events is not None and not df_pmt_events.empty:
            for section_min_ch2, section_max_ch2 in CH2_AGGREGATION_SECTIONS:
                df_segment_data_ch2_based = df_pmt_events[(df_pmt_events['ch2_pC'] >= section_min_ch2) &
                                                          (df_pmt_events['ch2_pC'] <= section_max_ch2)]
                x_center_ch2, y_mean_ch1, y_std_ch1 = plot_single_section_data_for_aggregation(
                    df_segment_data_ch2_based, pmt_full_name, (section_min_ch2, section_max_ch2)
                )
                if not np.isnan(y_mean_ch1) and not np.isnan(x_center_ch2):
                    aggregated_data_for_errorbar_plot.append({
                        'x_center': x_center_ch2, 'y_mean': y_mean_ch1, 'y_std': y_std_ch1
                    })

        plot_aggregated_errorbar_and_fit(ax_current_pmt_agg, pmt_full_name, aggregated_data_for_errorbar_plot)
        ax_current_pmt_agg.grid(True, linestyle=':', lw=0.5)
        ax_current_pmt_agg.tick_params(labelsize=12)

    fig_agg_overview.suptitle(f"Charge Correlation (Ch2 vs Ch1) - Aggregated Sections - {lom_name}", fontsize=18, y=1.0)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    overview_agg_output_filename = f"{lom_output_prefix}all_pmts_aggregated_curve_fits_ch2_vs_ch1_overview.pdf"
    plt.savefig(os.path.join(base_output_dir, lom_name, overview_agg_output_filename))
    plt.close(fig_agg_overview)
    print(f'Saved all PMTs Aggregated Curve Fits Overview for {lom_name} to: {os.path.join(base_output_dir, lom_name, overview_agg_output_filename)}')


def plot_single_charge_correlation_subplot(ax, df_pmt_events, pmt_full_name,
                                            fitted_segments_info,
                                            plot_type='log', cbar_vmin=None, cbar_vmax=None):
    """
    Plots a single PMT's 2D charge correlation histogram with piecewise curved fit.
    """
    x_data_all = df_pmt_events['ch1_pC'].values
    y_data_all = df_pmt_events['ch2_pC'].values

    x_plot_min_pC = 0
    x_plot_max_pC = 100
    y_plot_min_pC = 0
    y_plot_max_pC = 10

    x_bins_focused = np.linspace(x_plot_min_pC, x_plot_max_pC, 101)
    y_bins_focused = np.linspace(y_plot_min_pC, y_plot_max_pC, 51)

    if plot_type == 'log':
        hist_counts, _, _ = np.histogram2d(x_data_all, y_data_all, bins=[x_bins_focused, y_bins_focused])
        if np.max(hist_counts) == 0:
            h = ax.hist2d(x_data_all, y_data_all, bins=[x_bins_focused, y_bins_focused], cmap='viridis', vmin=cbar_vmin, vmax=cbar_vmax)
            title_suffix = '(Linear Scale - No Log Data)'
        else:
            h = ax.hist2d(x_data_all, y_data_all, bins=[x_bins_focused, y_bins_focused], cmap='viridis', norm=LogNorm(vmin=cbar_vmin, vmax=cbar_vmax))
            title_suffix = '(Log Scale)'
    else:
        h = ax.hist2d(x_data_all, y_data_all, bins=[x_bins_focused, y_bins_focused], cmap='viridis', vmin=cbar_vmin, vmax=cbar_vmax)
        title_suffix = '(Linear Scale)'
    
    ax.set_title(f'{pmt_full_name} {title_suffix}', fontsize=12)
    ax.set_xlabel('Charge Ch1 (pC)', fontsize=12)
    ax.set_ylabel('Charge Ch2 (pC)', fontsize=12)
    ax.set_xlim(x_plot_min_pC, x_plot_max_pC)
    ax.set_ylim(y_plot_min_pC, y_plot_max_pC)
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)

    if fitted_segments_info:
        x_combined_fit = []
        y_combined_fit = []
        r_squared_values = []

        for segment in fitted_segments_info:
            segment_min_x = segment['fit_min_x']
            segment_max_x = segment['fit_max_x']
            
            if 'poly_params' in segment and len(segment['poly_params']) == 3:
                poly_a, poly_b, poly_c = segment['poly_params']
                
                x_segment_plot = np.linspace(segment_min_x, segment_max_x, 50)
                y_segment_plot = polynomial_function(x_segment_plot, poly_a, poly_b, poly_c)
                
                valid_indices = (x_segment_plot >= x_plot_min_pC) & (x_segment_plot <= x_plot_max_pC) & \
                                (y_segment_plot >= y_plot_min_pC) & (y_segment_plot <= y_plot_max_pC)
                
                x_combined_fit.extend(x_segment_plot[valid_indices])
                y_combined_fit.extend(y_segment_plot[valid_indices])
                
                if 'r_squared' in segment and not np.isnan(segment['r_squared']):
                    r_squared_values.append(segment['r_squared'])
            else:
                warnings.warn(f"Skipping plot for segment [{segment_min_x:.1f}-{segment_max_x:.1f}] due to missing or invalid polynomial parameters.")

        if x_combined_fit:
            sorted_indices = np.argsort(x_combined_fit)
            ax.plot(np.array(x_combined_fit)[sorted_indices], np.array(y_combined_fit)[sorted_indices],
                    color='red', linestyle='--', linewidth=1.5)

            overall_r_squared_avg = np.nanmean(r_squared_values) if r_squared_values else np.nan
            
            legend_text = f'Piecewise Curved Fit\n(Avg $R^2={overall_r_squared_avg:.2f}$)'
            
            ax.legend(fontsize=12, loc='upper left', title=legend_text)
        else:
            ax.text(0.5, 0.5, "No valid piecewise curved fit data.", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=8, color='gray')

    else:
        ax.text(0.5, 0.5, "No valid piecewise fit data.", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=8, color='gray')

    return h[3], np.nan, np.nan, np.nan


def plot_multi_pmt_charge_correlations_overview(pmt_dfs_for_correlation, lom_name, base_output_dir, lom_output_prefix):
    """
    Orchestrates plotting Charge Correlation matrices for all 16 PMTs of a LOM.
    Generates:
    1. An overview PDF (Ch1 vs Ch2, 2D hist with piecewise CURVED fit based on CH1_OVERVIEW_SECTIONS).
    2. Individual PDFs for each PMT showing aggregated data (Ch2 vs Ch1 error bars) with a curved fit.
    3. Individual PDFs for each PMT showing Ch1 vs Ch2 scatter, colored by Ch1 value with section lines. (MODIFIED!)
    Also collects and exports fitting parameters to XLSX and a text summary.
    """
    LOG_CBAR_VMIN = 1e0
    LOG_CBAR_VMAX = 1e3

    fitting_params_data = []
    fitting_equations_summary = []

    custom_subplot_order = np.array([8, 10, 12, 14, 0, 2, 4, 6, 1, 3, 5, 7, 9, 11, 13, 15])
    pmt_ids_present = sorted(pmt_dfs_for_correlation.keys())
    pmt_ids_to_plot_ordered = [pid for pid in custom_subplot_order if pid in pmt_ids_present]
    if len(pmt_ids_to_plot_ordered) < 16:
        pmt_ids_to_plot_ordered = sorted(pmt_dfs_for_correlation.keys())

    main_mappable = None # Initialize main_mappable before the loop

    fitting_equations_summary.append(f"--- Fitting Equations Summary for {lom_name} (Piecewise Curved Regression) ---")

    lom_specific_output_dir = os.path.join(base_output_dir, lom_name)
    os.makedirs(lom_specific_output_dir, exist_ok=True)
    
    individual_aggregated_errorbar_output_dir = os.path.join(lom_specific_output_dir, "aggregated_curve_fits")
    os.makedirs(individual_aggregated_errorbar_output_dir, exist_ok=True)

    # Output directory for the new scatter plots (replacing 1D histograms)
    ch1_ch2_scatter_by_ch1_section_output_dir = os.path.join(lom_specific_output_dir, "ch1_ch2_scatter_by_ch1_section")
    os.makedirs(ch1_ch2_scatter_by_ch1_section_output_dir, exist_ok=True)


    fig_overview, axes_overview = plt.subplots(4, 4, figsize=(20, 16))
    axes_overview = axes_overview.flatten()


    for i, pmt_id in enumerate(pmt_ids_to_plot_ordered):
        ax_current_pmt_overview = axes_overview[i]
        df_pmt_events = pmt_dfs_for_correlation.get(pmt_id)
        pmt_full_name = f"PMT {pmt_id:02d}"

        fitted_segments_for_overview_plot = []
        all_segment_r_squareds = []
        if df_pmt_events is not None and not df_pmt_events.empty:
            for section_min_x_ch1, section_max_x_ch1 in CH1_OVERVIEW_SECTIONS:
                df_segment_data_ch1_based = df_pmt_events[(df_pmt_events['ch1_pC'] >= section_min_x_ch1) &
                                                          (df_pmt_events['ch1_pC'] <= section_max_x_ch1)]
                if len(df_segment_data_ch1_based) >= 3:
                    try:
                        params_local, cov_local = curve_fit(polynomial_function,
                                                            df_segment_data_ch1_based['ch1_pC'].values,
                                                            df_segment_data_ch1_based['ch2_pC'].values,
                                                            p0=[0.001, 0.05, 0.5])
                        
                        y_pred_local = polynomial_function(df_segment_data_ch1_based['ch1_pC'].values, *params_local)
                        residuals_local = df_segment_data_ch1_based['ch2_pC'].values - y_pred_local
                        ss_res_local = np.sum(residuals_local**2)
                        ss_tot_local = np.sum((df_segment_data_ch1_based['ch2_pC'].values - np.mean(df_segment_data_ch1_based['ch2_pC'].values))**2)
                        r_squared_local = 1 - (ss_res_local / ss_tot_local) if ss_tot_local > 0 else np.nan
                        
                        fitted_segments_for_overview_plot.append({
                            'fit_min_x': section_min_x_ch1, 'fit_max_x': section_max_x_ch1,
                            'poly_params': params_local.tolist(),
                            'r_squared': r_squared_local
                        })
                        if not np.isnan(r_squared_local): all_segment_r_squareds.append(r_squared_local)
                    except Exception as e:
                        warnings.warn(f"Could not fit polynomial regression for {pmt_full_name} Ch1 section [{section_min_x_ch1:.1f}-{section_max_x_ch1:.1f}]: {e}.")
                else:
                    warnings.warn(f"Not enough data points ({len(df_segment_data_ch1_based)}) for polynomial fit in {pmt_full_name} Ch1 section [{section_min_x_ch1:.1f}-{section_max_x_ch1:.1f}]. Requires at least 3 for degree 2 polynomial.")
        else: warnings.warn(f"No events data for {pmt_full_name}. Skipping overview and aggregated plots.")

        # Assign mappable here, ensuring it's always assigned if a plot is generated
        current_mappable, _, _, _ = plot_single_charge_correlation_subplot(
            ax_current_pmt_overview, df_pmt_events, pmt_full_name,
            fitted_segments_info=fitted_segments_for_overview_plot,
            plot_type='log', cbar_vmin=LOG_CBAR_VMIN, cbar_vmax=LOG_CBAR_VMAX
        )
        if main_mappable is None:
            main_mappable = current_mappable # Assign the first valid mappable


        aggregated_data_for_errorbar_plot = []
        if df_pmt_events is not None and not df_pmt_events.empty:
            for section_min_ch2, section_max_ch2 in CH2_AGGREGATION_SECTIONS:
                df_segment_data_ch2_based = df_pmt_events[(df_pmt_events['ch2_pC'] >= section_min_ch2) &
                                                          (df_pmt_events['ch2_pC'] <= section_max_ch2)]
                x_center_ch2, y_mean_ch1, y_std_ch1 = plot_single_section_data_for_aggregation(
                    df_segment_data_ch2_based, pmt_full_name, (section_min_ch2, section_max_ch2)
                )
                if not np.isnan(y_mean_ch1) and not np.isnan(x_center_ch2):
                    aggregated_data_for_errorbar_plot.append({
                        'x_center': x_center_ch2, 'y_mean': y_mean_ch1, 'y_std': y_std_ch1
                    })

            fig_individual_agg, ax_individual_agg = plt.subplots(1, 1, figsize=(8, 6))
            plot_aggregated_errorbar_and_fit(
                ax_individual_agg,
                pmt_full_name,
                aggregated_data_for_errorbar_plot
            )
            individual_output_filename = f"{pmt_full_name.replace(' ', '')}_Aggregated_Curve_Fit_Ch2_vs_Ch1.pdf"
            plt.savefig(os.path.join(individual_aggregated_errorbar_output_dir, individual_output_filename))
            plt.close(fig_individual_agg)
            print(f'Saved individual aggregated curve fit plot for {pmt_full_name} to: {os.path.join(individual_aggregated_errorbar_output_dir, individual_output_filename)}')
        
        # --- CALLING THE NEW SCATTER PLOT FUNCTION HERE ---
        if df_pmt_events is not None and not df_pmt_events.empty:
            plot_ch1_ch2_scatter_by_ch1_color(df_pmt_events, pmt_full_name, ch1_ch2_scatter_by_ch1_section_output_dir)


        overall_r_squared_avg = np.nanmean(all_segment_r_squareds) if all_segment_r_squareds else np.nan
        fitting_params_data.append({
            'LOM ID': lom_name,
            'PMT ID': pmt_id,
            'Fit Type': 'Piecewise Curved (Ch1 vs Ch2 - Polynomial Degree 2)',
            'Overall Avg R-squared': overall_r_squared_avg,
            'Segment Details': json.dumps(fitted_segments_for_overview_plot)
        })
        if not np.isnan(overall_r_squared_avg):
            eq_str = f"Piecewise Curved Fit (Avg $R^2={overall_r_squared_avg:.4f}$)"
            fitting_equations_summary.append(f"{lom_name} PMT {pmt_id:02d}: {eq_str}")
        else:
            fitting_equations_summary.append(f"{lom_name} PMT {pmt_id:02d}: No valid piecewise curved fit.")

        ax_current_pmt_overview.grid(True, linestyle=':', lw=0.5)
        ax_current_pmt_overview.tick_params(labelsize=12)

    if main_mappable is not None:
        cbar_ax_overview = fig_overview.add_axes([0.92, 0.1, 0.02, 0.8])
        fig_overview.colorbar(main_mappable, cax=cbar_ax_overview, label='Number of Events (Log Scale)', format='%.0e', ticks=[10**i for i in range(int(np.log10(LOG_CBAR_VMIN)), int(np.log10(LOG_CBAR_VMAX)) + 1)])
    else:
        warnings.warn(f"No valid mappable found for the main overview plot colorbar for LOM {lom_name}. Colorbar will not be generated.")


    fig_overview.suptitle(f"Charge Correlation Overview (Ch1 vs Ch2, Piecewise Curved Fit) - {lom_name}", fontsize=18, y=1.0)
    plt.tight_layout(rect=[0, 0.03, 0.9, 0.96])
    
    overview_output_filename = f"{lom_output_prefix}all_pmts_charge_correlation_piecewise_curved_overview.pdf"
    plt.savefig(os.path.join(lom_specific_output_dir, overview_output_filename))
    plt.close(fig_overview)
    print(f'Saved all PMTs Charge Correlation Piecewise Curved Overview for {lom_name} to: {os.path.join(lom_specific_output_dir, overview_output_filename)}')

    df_fitting_params = pd.DataFrame(fitting_params_data)
    xlsx_filename = f"{lom_output_prefix}charge_correlation_piecewise_curved_fitting_params.xlsx"
    xlsx_path = os.path.join(lom_specific_output_dir, xlsx_filename)
    df_fitting_params.to_excel(xlsx_path, index=False)
    print(f"Exported Charge Correlation Piecewise Curved Fitting Parameters for {lom_name} to: {xlsx_path}")

    txt_filename = f"{lom_output_prefix}charge_correlation_piecewise_curved_equations_summary.txt"
    txt_path = os.path.join(lom_specific_output_dir, txt_filename)
    with open(txt_path, 'w') as f:
        for line in fitting_equations_summary:
            f.write(line + '\n')
    print(f"Exported Charge Correlation Piecewise Curved Equations Summary for {lom_name} to: {txt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Charge Correlation plots with piecewise curved fits and aggregated error bar plots for multiple LOMs.")
    parser.add_argument("lom_data_dirs", nargs='+', help="Input LOM data directories (e.g., 'path/to/LOM16-01/'). Each directory should contain 16 PMT HDF files.", type=str)
    parser.add_argument("--output_dir", "-o", help="Base output directory for all LOM plots (e.g., './my_analysis_results'). Each LOM will have its own subdirectory within this.", type=str, default='./charge_correlation_results')
    
    args = parser.parse_args()

    output_base_dir_for_all_loms = args.output_dir
    os.makedirs(output_base_dir_for_all_loms, exist_ok=True)
    print(f"All LOM plots will be organized within: {output_base_dir_for_all_loms}")

    for current_lom_folder_path in args.lom_data_dirs:
        current_lom_folder_path = os.path.normpath(current_lom_folder_path)
        current_lom_name = os.path.basename(current_lom_folder_path)

        print(f"\n========================================================")
        print(f"=== Processing LOM: {current_lom_name} ===")
        print(f"========================================================")

        pmt_to_infile_map = {}
        current_lom_runid = None

        current_lom_pmt_files = glob.glob(os.path.join(current_lom_folder_path, "*.hdf"))
        current_lom_pmt_files.sort()

        if not current_lom_pmt_files:
            warnings.warn(f"Error: No HDF files found directly in LOM folder '{current_lom_folder_path}'. Skipping this LOM.")
            continue

        for fname in current_lom_pmt_files:
            file_base = os.path.basename(fname)
            file_base_no_ext = os.path.splitext(file_base)[0]

            pmt_id = -1
            pmt_id_match = re.search(r'\.(\d{2})$', file_base_no_ext)
            if pmt_id_match:
                pmt_id = int(pmt_id_match.group(1))

            if 0 <= pmt_id <= 15:
                pmt_to_infile_map[pmt_id] = fname
            else:
                warnings.warn(f"Parsed PMT ID {pmt_id} from filename '{fname}' is out of expected range (0-15) or invalid. Skipping for PMT map for LOM {current_lom_name}.")

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
            warnings.warn(f"Warning: Only {len(pmt_to_infile_map)} PMT files found for LOM {current_lom_name}. Plot may be incomplete.")

        lom_output_prefix = current_lom_name
        if current_lom_runid is not None:
            lom_output_prefix = f"{current_lom_name}_Run{current_lom_runid}_"
        else:
            lom_output_prefix = f"{current_lom_name}_"


        pmt_dfs_for_correlation_overview = {}

        print(f"\n--- Collecting data for PMT-specific plots in {current_lom_name} ---")
        pmt_ids_with_data_sorted = sorted(pmt_to_infile_map.keys())

        for pmt_id in pmt_ids_with_data_sorted:
            infile_pmt = pmt_to_infile_map[pmt_id]
            pmt_events_data = []
            data_pmt = read_hdffile(infile_pmt)
            
            if data_pmt and "data" in data_pmt and "metadata" in data_pmt:
                required_data_keys = ["charge_ch1", "charge_ch2", "FPGAtime"]
                required_metadata_keys = ["conversion_ch1", "conversion_ch2"]
                
                data_present = all(key in data_pmt["data"] and data_pmt["data"][key] is not None for key in required_data_keys)
                metadata_present = all(key in data_pmt["metadata"] and data_pmt["metadata"][key] is not None for key in required_metadata_keys)

                if data_present and metadata_present:
                    try:
                        raw_charges_ch1_adc = data_pmt["data"]["charge_ch1"]
                        conversion_ch1 = data_pmt["metadata"]["conversion_ch1"]
                        raw_charges_ch2_adc = data_pmt["data"]["charge_ch2"]
                        conversion_ch2 = data_pmt["metadata"]["conversion_ch2"]
                        fpga_time = data_pmt["data"]["FPGAtime"]

                        if isinstance(conversion_ch1, np.ndarray) and conversion_ch1.ndim == 0:
                            conversion_ch1 = conversion_ch1.item()
                        if isinstance(conversion_ch2, np.ndarray) and conversion_ch2.ndim == 0:
                            conversion_ch2 = conversion_ch2.item()

                        conversion_factor_pC = (1e-6 * (1/60e6) * 1e12)
                        
                        raw_charges_ch1_adc = np.asarray(raw_charges_ch1_adc)
                        raw_charges_ch2_adc = np.asarray(raw_charges_ch2_adc)
                        fpga_time = np.asarray(fpga_time)

                        charges_ch1_pC = raw_charges_ch1_adc * (conversion_ch1 * conversion_factor_pC)
                        charges_ch2_pC = raw_charges_ch2_adc * (conversion_ch2 * conversion_factor_pC)

                        min_len_pmt = min(len(charges_ch1_pC), len(charges_ch2_pC), len(fpga_time))
                        for k in range(min_len_pmt):
                            pmt_events_data.append({
                                'ch1_pC': charges_ch1_pC[k],
                                'ch2_pC': charges_ch2_pC[k],
                                'filename': infile_pmt,
                                'timestamp': fpga_time[k],
                                'event_idx': k,
                                'pmt_id': pmt_id
                            })
                    except Exception as e:
                        warnings.warn(f"Error processing data for PMT {pmt_id} from '{infile_pmt}': {e}. Skipping for correlation.")
                else:
                    missing_data = [key for key in required_data_keys if key not in data_pmt["data"] or data_pmt["data"][key] is None]
                    missing_metadata = [key for key in required_metadata_keys if key not in data_pmt["metadata"] or data_pmt["metadata"][key] is None]
                    
                    error_msg = f"Missing data/metadata keys for PMT {pmt_id} in '{infile_pmt}'. "
                    if missing_data: error_msg += f"Missing data: {', '.join(missing_data)}. "
                    if missing_metadata: error_msg += f"Missing metadata: {', '.join(missing_metadata)}. "
                    warnings.warn(error_msg + "Skipping for correlation.")
            else:
                warnings.warn(f"Missing 'data' or 'metadata' group in '{infile_pmt}'. Skipping for correlation.")
            
            df_pmt_events = pd.DataFrame(pmt_events_data)
            
            if not df_pmt_events.empty and all(col in df_pmt_events.columns for col in ['ch1_pC', 'ch2_pC']):
                df_pmt_events = df_pmt_events.dropna(subset=['ch1_pC', 'ch2_pC'])
                df_pmt_events = df_pmt_events[(df_pmt_events['ch1_pC'] > 1e-3) & (df_pmt_events['ch2_pC'] > 1e-3)]
            else:
                warnings.warn(f"DataFrame for PMT {pmt_id} is empty or missing 'ch1_pC'/'ch2_pC' after data loading. Skipping analysis for this PMT.")
                df_pmt_events = pd.DataFrame()

            pmt_dfs_for_correlation_overview[pmt_id] = df_pmt_events.copy()

        print(f"\n--- Generating Charge Correlation Plots for LOM {current_lom_name} ---")

        plot_multi_pmt_charge_correlations_overview(
            pmt_dfs_for_correlation_overview,
            current_lom_name,
            output_base_dir_for_all_loms,
            lom_output_prefix
        )

        print(f"\n--- Generating Combined Aggregated Curve Fit Plot for LOM {current_lom_name} ---")
        plot_multi_pmt_aggregated_curve_fits_overview(
            pmt_dfs_for_correlation_overview,
            current_lom_name,
            output_base_dir_for_all_loms,
            lom_output_prefix
        )


    print("\n========================================================")
    print("=== Charge Correlation Plotting Complete. ===")
    print("========================================================")