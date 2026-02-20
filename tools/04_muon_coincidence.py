# full_muon_analysis.py
#
# Comprehensive Muon Analysis Script:
# - Reads HDF files (Muon or SPE data).
# - Plots Charge Distributions in pC and Photoelectrons (PE) with logarithmic Y-axis.
# - Generates PMT vs Coincidence Rate plot.
# - Generates Coincidence Matrix (heatmap) for pairwise coincidences.
# - Plots raw waveforms of selected coincident events.
# - Calculates livetime based on FPGA timestamps.
# - Allows specifying an output directory for all saved plots.
#
import h5py
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import glob
import sys, os, json
import argparse

# External libraries that were in supervisor's sample script but might not be explicitly used in this combined logic
from lmfit import Model
from matplotlib.colors import LogNorm
from scipy.interpolate import UnivariateSpline

# === START: ส่วนการจัดการ PATH และ IMPORT ที่ถูกต้องและแข็งแกร่งขึ้น ===
script_dir = os.path.dirname(os.path.abspath(__file__))
util_path = os.path.join(script_dir, "util")
if util_path not in sys.path:
    sys.path.append(util_path)

from HDFWriterModuleInspection import load_dict
from eventHist import * # Contains plot_charge
from plotting_functions import * # Contains plot_fpga_time_delta (though not explicitly used for plotting here)
# read_hdffile is imported from HDFWriterModuleInspection via load_dict if not defined
try:
    read_hdffile
except NameError:
    read_hdffile = load_dict
# === END: ส่วนการจัดการ PATH และ IMPORT ที่ถูกต้องและแข็งแกร่งขึ้น ===


plotting_map = [6,11,7,12,8,13,9,14,1,16,2,17,3,18,4,19,5,20] # For 20 PMTs in 4x5 grid


# --- Start: Bug-fixed Helper Functions (from muon_analysis_coincidence_script.py) ---

def get_timestamps(filename, min_charge_pC=0.0, max_charge_pC=float('inf')):
    """
    Retrieves FPGA timestamps and event indices for events
    within a specified charge range from a specific HDF file.
    """
    ret_timestamps=[]
    ret_eventidx=[]
    
    data = read_hdffile(filename, quiet=True) 
            
    q_ch1 = data["data"]["charge_ch1"][()]
    conversion_ch1 = data["metadata"]["conversion_ch1"][()]
    q_ch1_pC = np.array([ s * (conversion_ch1* 1e-6 * (1/60e6) * 1e12) for s in q_ch1])

    fpga_time = data["data"]["FPGAtime"][()]      
    fpga_time = np.array(fpga_time)

    for iev, (q,t) in enumerate( zip(q_ch1_pC, fpga_time) ):
        if min_charge_pC <= q <= max_charge_pC:
            ret_timestamps.append(t)
            ret_eventidx.append(iev)

    return ret_timestamps, ret_eventidx


def event_matching(timestamp_ref_ns, timestamp_ns, window_ns=100.):
    """
    Finds matching timestamps between a reference set and a target set
    within a specified time window. Uses a two-pointer approach assuming sorted timestamps.
    Returns matched pairs and count of matches.
    """
    ret_times=[]
    ngood=0

    ptr_ref, ptr_target = 0, 0
    while ptr_ref < len(timestamp_ref_ns) and ptr_target < len(timestamp_ns):
        t_ref = timestamp_ref_ns[ptr_ref]
        t_target = timestamp_ns[ptr_target]
        
        if abs(t_ref - t_target) <= window_ns:
            ret_times.append([t_ref, t_target])
            ngood += 1
            ptr_ref += 1
            ptr_target += 1
        elif t_ref > t_target:
            ptr_target += 1
        else: # t_target > t_ref
            ptr_ref += 1
    
    return ret_times, ngood


def get_waveform_at_this_timestamp(filename, timestamp):
    """
    Retrieves the raw waveform (ADC counts) for a specific timestamp
    from a given HDF file.
    """
    data = read_hdffile(filename, quiet=True) 

    fpga_time = data["data"]["FPGAtime"][()]      
    fpga_time = np.array(fpga_time)

    nsamples = data["data"]["nsample"][()]
    adc_ch1 = data["data"]["ADC_ch1"][()]
    adc_ch2 = data["data"]["ADC_ch2"][()]    

    x = []
    wf_ch1 = []
    wf_ch2 = []

    matching_indices = np.where(fpga_time == timestamp)[0]
    if len(matching_indices) > 0:
        iev = matching_indices[0] # Take the first match
        n = nsamples[iev]
        x = [ i *(1e9/60e6) for i in range(n) ] # Convert samples to ns
        wf_ch1 = adc_ch1[iev][:n]
        wf_ch2 = adc_ch2[iev][:n]
        
    return x, wf_ch1, wf_ch2

def get_charges_of_these_events(filename, evidx): 
    """
    Retrieves charge values for specific event indices from an HDF file.
    """
    ret_charges=[]

    data = read_hdffile(filename, quiet=True) 

    q_ch1 = data["data"]["charge_ch1"][()]
    conversion_ch1 = data["metadata"]["conversion_ch1"][()]
    q_ch1_pC = [ s * (conversion_ch1* 1e-6 * (1/60e6) * 1e12) for s in q_ch1] 

    for iev, q in enumerate(q_ch1_pC):
        if iev in evidx:
            ret_charges.append(q)

    return ret_charges
# --- End: Bug-fixed Helper Functions ---


# --- Start: Plotting Functions (from analysis_quick_check.py, with Log-scale and output_dir) ---
def plot_all_charge_dists(filenames, xlims, unit_label='Charge (pC)', save_suffix='ch1', output_dir='./figs'):
    """
    Plots charge distributions for all PMTs in pC with logarithmic Y-axis.
    Files are saved to the specified output_dir.
    """
    fig_ch1 = plt.figure(figsize=(12, 9))
    fig_ch1_wide = plt.figure(figsize=(12, 9))
    fig_ch2 = plt.figure(figsize=(12, 9))

    for infile in filenames:
        if not os.path.isfile(infile):
            print(f"File not found or path is incorrect: {infile}. Skipping.")
            continue

        base_filename = os.path.basename(infile)
        try:
            ch_str_parts = base_filename.split(".")
            if len(ch_str_parts) >= 2:
                potential_ch_part = ch_str_parts[-2]
                if '_' in potential_ch_part:
                    ch = int(potential_ch_part.split('_')[-1])
                else: 
                    ch = int(potential_ch_part)
            else:
                ch = int(base_filename.split("_")[-1].split(".")[0]) 

        except (ValueError, IndexError):
            print(f"Could not parse channel number from filename: {infile}. Skipping.")
            continue

        if ch < 0 or ch >= len(plotting_map):
            print(f"Warning: Channel number {ch} from file {infile} is out of plotting_map range (0-{len(plotting_map)-1}). Skipping plot for this channel.")
            continue
        subplot_index = plotting_map[ch]
        if subplot_index < 1 or subplot_index > 20:
            print(f"Warning: Subplot index {subplot_index} for channel {ch} is out of valid range (1-20). Skipping plot for this channel.")
            continue

        data = read_hdffile(infile, quiet=True)

        ch1, ch2 = plot_charge(data, no_plot=True, xmin_ch1=-1.0, xmax_ch1=2.0, nbin_ch1=114*3)
        ch1_wide, ch2_wide = plot_charge(data, no_plot=True, xmin_ch1=-1, xmax_ch1=200, nbin_ch1=114*3)

        # Plotting for ch1 (narrow range)
        ax_ch1 = fig_ch1.add_subplot(4,5, subplot_index)
        ax_ch1.plot(ch1[0], ch1[1], lw=1., label=f"pmt{ch}", color='blue')
        ax_ch1.set_xticks([0.0, 0.8*0.2, 0.8])
        ylims = ax_ch1.get_ylim()
        ax_ch1.vlines(x=0.8, ymin=ylims[0], ymax=ylims[1], linestyle=":", colors='red', lw=0.5)
        ax_ch1.vlines(x=0.8*0.8, ymin=ylims[0], ymax=ylims[1], linestyle=":", colors='orange', lw=0.5)
        ax_ch1.vlines(x=0.8*1.2, ymin=ylims[0], ymax=ylims[1], linestyle=":", colors='orange', lw=0.5)
        if xlims: ax_ch1.set_xlim([xlims[0],xlims[1]])
        ax_ch1.grid('xy', linestyle=':', lw=0.5)
        ax_ch1.tick_params(labelsize=6)
        ax_ch1.set_yscale('log')
        if subplot_index in [5, 17,18,19,20]:
            ax_ch1.tick_params(axis='y', length=0)
            ax_ch1.set_yticklabels([])
            ax_ch1.set_xlabel(unit_label,fontsize=6)
        elif subplot_index in [ 1,6,11]:
            ax_ch1.tick_params(axis='x', length=0)
            ax_ch1.set_xticklabels([])
        elif subplot_index==16:
            ax_ch1.set_xlabel(unit_label,fontsize=6)
        else:
            ax_ch1.tick_params(axis='both', length=0)
            ax_ch1.set_yticklabels([])
            ax_ch1.set_xticklabels([])
        ax_ch1.legend(fontsize=6, loc='upper right')

        # Plotting for ch2 (narrow range)
        ax_ch2 = fig_ch2.add_subplot(4,5, subplot_index)
        ax_ch2.plot(ch2[0], ch2[1], lw=1., label=f"pmt{ch}", color='green')
        ax_ch2.grid('xy', linestyle=':', lw=0.5)
        ax_ch2.tick_params(labelsize=6)
        if subplot_index in [5, 17,18,19,20]:
            ax_ch2.tick_params(axis='y', length=0)
            ax_ch2.set_yticklabels([])
            ax_ch2.set_xlabel(unit_label,fontsize=6)
        elif subplot_index in [ 1,6,11]:
            ax_ch2.tick_params(axis='x', length=0)
            ax_ch2.set_xticklabels([])
        elif subplot_index==16:
            ax_ch2.set_xlabel(unit_label,fontsize=6)
        else:
            ax_ch2.tick_params(axis='both', length=0)
            ax_ch2.set_yticklabels([])
            ax_ch2.set_xticklabels([])
        ax_ch2.legend(fontsize=6)

        # Plotting for ch1 (wide range)
        ax_ch1_wide = fig_ch1_wide.add_subplot(4,5, subplot_index)
        ax_ch1_wide.plot(ch1_wide[0], ch1_wide[1], lw=1., label=f"pmt{ch}", color='blue')
        ax_ch1_wide.set_xlim([-1,200])
        ax_ch1_wide.grid('xy', linestyle=':', lw=0.5)
        ax_ch1_wide.tick_params(labelsize=6)
        ax_ch1_wide.set_yscale('log')
        if subplot_index in [5, 17,18,19,20]:
            ax_ch1_wide.tick_params(axis='y', length=0)
            ax_ch1_wide.set_yticklabels([])
            ax_ch1_wide.set_xlabel(unit_label,fontsize=6)
        elif subplot_index in [ 1,6,11]:
            ax_ch1_wide.tick_params(axis='x', length=0)
            ax_ch1_wide.set_xticklabels([])
        elif subplot_index==16:
            ax_ch1_wide.set_xlabel(unit_label,fontsize=6)
        else:
            ax_ch1_wide.tick_params(axis='both', length=0)
            ax_ch1_wide.set_yticklabels([])
            ax_ch1_wide.set_xticklabels([])
        ax_ch1_wide.legend(fontsize=6, loc='upper right')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig_ch1.subplots_adjust(wspace=0, hspace=0)
    fig_ch1.savefig(f'{output_dir}/all_charge_{save_suffix}.pdf')
    fig_ch2.subplots_adjust(wspace=0, hspace=0)
    fig_ch2.savefig(f'{output_dir}/all_charge_{save_suffix}_ch2.pdf')
    fig_ch1_wide.subplots_adjust(wspace=0, hspace=0)
    fig_ch1_wide.savefig(f'{output_dir}/all_charge_{save_suffix}_wide.pdf')
    plt.close(fig_ch1)
    plt.close(fig_ch2)
    plt.close(fig_ch1_wide)
    print(f'Saved {output_dir}/all_charge_{save_suffix}.pdf, {output_dir}/all_charge_{save_suffix}_ch2.pdf, {output_dir}/all_charge_{save_suffix}_wide.pdf')


def plot_all_charge_dists_pe(filenames, assumed_gain=5e6, save_suffix='pe', output_dir='./figs'):
    """
    Plots charge distributions converted to Photoelectrons (PE) for all PMTs with logarithmic Y-axis.
    Assumes 1 PE charge = assumed_gain * 1.602e-19 C.
    Files are saved to the specified output_dir.
    """
    # Calculate conversion factor from pC to PE
    electron_charge_C = 1.602e-19 # Coulombs per electron
    charge_per_pe_C = assumed_gain * electron_charge_C # Coulombs per PE
    charge_per_pe_pC = charge_per_pe_C * 1e12 # pC per PE (1 pC = 1e-12 C)

    if charge_per_pe_pC == 0:
        print("Error: Charge per PE is zero. Cannot convert to PE. Check assumed_gain.")
        return

    # Call the generic plotting function with PE conversion
    # The x-range for PE should be determined based on the pC range divided by the conversion factor
    # For -0.1 to 1.5 pC -> -0.1/0.801 to 1.5/0.801 PE
    # For -1 to 200 pC -> -1/0.801 to 200/0.801 PE
    
    # Original pC xlims: [-0.1, 1.5]
    xlims_pe_narrow = [-0.1 / charge_per_pe_pC, 1.5 / charge_per_pe_pC]
    # Original pC xlims: [-1, 200]
    xlims_pe_wide = [-1 / charge_per_pe_pC, 200 / charge_per_pe_pC]


    fig_ch1 = plt.figure(figsize=(12, 9))
    fig_ch1_wide = plt.figure(figsize=(12, 9))
    fig_ch2 = plt.figure(figsize=(12, 9))

    for infile in filenames:
        if not os.path.isfile(infile):
            print(f"File not found or path is incorrect: {infile}. Skipping.")
            continue

        base_filename = os.path.basename(infile)
        try:
            ch_str_parts = base_filename.split(".")
            if len(ch_str_parts) >= 2:
                potential_ch_part = ch_str_parts[-2]
                if '_' in potential_ch_part:
                    ch = int(potential_ch_part.split('_')[-1])
                else:
                    ch = int(potential_ch_part)
            else:
                ch = int(base_filename.split("_")[-1].split(".")[0])

        except (ValueError, IndexError):
            print(f"Could not parse channel number from filename: {infile}. Skipping.")
            continue

        if ch < 0 or ch >= len(plotting_map):
            print(f"Warning: Channel number {ch} from file {infile} is out of plotting_map range (0-{len(plotting_map)-1}). Skipping plot for this channel.")
            continue
        subplot_index = plotting_map[ch]
        if subplot_index < 1 or subplot_index > 20:
            print(f"Warning: Subplot index {subplot_index} for channel {ch} is out of valid range (1-20). Skipping plot for this channel.")
            continue

        data = read_hdffile(infile, quiet=True)

        # Get charge in pC first
        ch1_pC, ch2_pC = plot_charge(data, no_plot=True, xmin_ch1=-1.0, xmax_ch1=2.0, nbin_ch1=114*3)
        ch1_wide_pC, ch2_wide_pC = plot_charge(data, no_plot=True, xmin_ch1=-1, xmax_ch1=200, nbin_ch1=114*3)

        # Convert x-axis (charge) from pC to PE
        ch1_pe = [np.array(ch1_pC[0]) / charge_per_pe_pC, ch1_pC[1]]
        ch2_pe = [np.array(ch2_pC[0]) / charge_per_pe_pC, ch2_pC[1]]
        ch1_wide_pe = [np.array(ch1_wide_pC[0]) / charge_per_pe_pC, ch1_wide_pC[1]]


        # Plotting for ch1 (narrow range, PE)
        ax_ch1 = fig_ch1.add_subplot(4,5, subplot_index)
        ax_ch1.plot(ch1_pe[0], ch1_pe[1], lw=1., label=f"pmt{ch}", color='blue')
        # Adjust x-ticks for PE: peak at 0.801 pC means 1 PE
        ax_ch1.set_xticks([0.0, 0.801*0.2/charge_per_pe_pC, 0.801/charge_per_pe_pC])
        ylims = ax_ch1.get_ylim()
        ax_ch1.vlines(x=0.801/charge_per_pe_pC, ymin=ylims[0], ymax=ylims[1], linestyle=":", colors='red', lw=0.5)
        ax_ch1.vlines(x=0.801*0.8/charge_per_pe_pC, ymin=ylims[0], ymax=ylims[1], linestyle=":", colors='orange', lw=0.5)
        ax_ch1.vlines(x=0.801*1.2/charge_per_pe_pC, ymin=ylims[0], ymax=ylims[1], linestyle=":", colors='orange', lw=0.5)
        ax_ch1.set_xlim([xlims_pe_narrow[0],xlims_pe_narrow[1]]) # Use PE-converted xlims
        ax_ch1.grid('xy', linestyle=':', lw=0.5)
        ax_ch1.tick_params(labelsize=6)
        ax_ch1.set_yscale('log')
        if subplot_index in [5, 17,18,19,20]:
            ax_ch1.tick_params(axis='y', length=0)
            ax_ch1.set_yticklabels([])
            ax_ch1.set_xlabel('Photoelectrons (PE)',fontsize=6)
        elif subplot_index in [ 1,6,11]:
            ax_ch1.tick_params(axis='x', length=0)
            ax_ch1.set_xticklabels([])
        elif subplot_index==16:
            ax_ch1.set_xlabel('Photoelectrons (PE)',fontsize=6)
        else:
            ax_ch1.tick_params(axis='both', length=0)
            ax_ch1.set_yticklabels([])
            ax_ch1.set_xticklabels([])
        ax_ch1.legend(fontsize=6, loc='upper right')

        # Plotting for ch2 (narrow range, PE)
        ax_ch2 = fig_ch2.add_subplot(4,5, subplot_index)
        ax_ch2.plot(ch2_pe[0], ch2_pe[1], lw=1., label=f"pmt{ch}", color='green')
        ax_ch2.grid('xy', linestyle=':', lw=0.5)
        ax_ch2.tick_params(labelsize=6)
        if subplot_index in [5, 17,18,19,20]:
            ax_ch2.tick_params(axis='y', length=0)
            ax_ch2.set_yticklabels([])
            ax_ch2.set_xlabel('Photoelectrons (PE)',fontsize=6)
        elif subplot_index in [ 1,6,11]:
            ax_ch2.tick_params(axis='x', length=0)
            ax_ch2.set_xticklabels([])
        elif subplot_index==16:
            ax_ch2.set_xlabel('Photoelectrons (PE)',fontsize=6)
        else:
            ax_ch2.tick_params(axis='both', length=0)
            ax_ch2.set_yticklabels([])
            ax_ch2.set_xticklabels([])
        ax_ch2.legend(fontsize=6)

        # Plotting for ch1 (wide range, PE)
        ax_ch1_wide = fig_ch1_wide.add_subplot(4,5, subplot_index)
        ax_ch1_wide.plot(ch1_wide_pe[0], ch1_wide_pe[1], lw=1., label=f"pmt{ch}", color='blue')
        ax_ch1_wide.set_xlim([xlims_pe_wide[0], xlims_pe_wide[1]]) # Use PE-converted wide xlims
        ax_ch1_wide.grid('xy', linestyle=':', lw=0.5)
        ax_ch1_wide.tick_params(labelsize=6)
        ax_ch1_wide.set_yscale('log')
        if subplot_index in [5, 17,18,19,20]:
            ax_ch1_wide.tick_params(axis='y', length=0)
            ax_ch1_wide.set_yticklabels([])
            ax_ch1_wide.set_xlabel('Photoelectrons (PE)',fontsize=6)
        elif subplot_index in [ 1,6,11]:
            ax_ch1_wide.tick_params(axis='x', length=0)
            ax_ch1_wide.set_xticklabels([])
        elif subplot_index==16:
            ax_ch1_wide.set_xlabel('Photoelectrons (PE)',fontsize=6)
        else:
            ax_ch1_wide.tick_params(axis='both', length=0)
            ax_ch1_wide.set_yticklabels([])
            ax_ch1_wide.set_xticklabels([])
        ax_ch1_wide.legend(fontsize=6, loc='upper right')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig_ch1.subplots_adjust(wspace=0, hspace=0)
    fig_ch1.savefig(f'{output_dir}/all_charge_{save_suffix}.pdf')
    fig_ch2.subplots_adjust(wspace=0, hspace=0)
    fig_ch2.savefig(f'{output_dir}/all_charge_{save_suffix}_ch2.pdf')
    fig_ch1_wide.subplots_adjust(wspace=0, hspace=0)
    fig_ch1_wide.savefig(f'{output_dir}/all_charge_{save_suffix}_wide.pdf')
    plt.close(fig_ch1)
    plt.close(fig_ch2)
    plt.close(fig_ch1_wide)
    print(f'Saved {output_dir}/all_charge_{save_suffix}.pdf, {output_dir}/all_charge_{save_suffix}_ch2.pdf, {output_dir}/all_charge_{save_suffix}_wide.pdf')
# --- End: Plotting Functions ---


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive Muon Analysis Script: Combines charge distribution plotting and coincidence analysis.")
    parser.add_argument("infiles", nargs='+', help="Input HDF file names (e.g., 'muon_data/data_muon_run_*.hdf')", type=str)
    parser.add_argument("--output_dir", "-o", help="Output directory for plots (default: ./plots_output)", type=str, default='./plots_output') # New default output dir for merged script
    args = parser.parse_args()

    # --- Start: File Scanning and PMT Map Creation (from muon_analysis_coincidence_script.py) ---
    infilenames = []
    for f in args.infiles:
        expanded_files = glob.glob(f) # Expand wildcards
        for infile_path in expanded_files:
            if os.path.isfile(infile_path) and infile_path.lower().endswith(".hdf"):
                infilenames.append(infile_path)

    if not infilenames: 
        print("Error: No HDF files found matching the pattern. Please check input path and filename extensions.")
        sys.exit(1)

    print(f"Found {len(infilenames)} files: {infilenames}")

    runid=None
    lomname=None
    pmt_to_infile_map = {}

    for fname in infilenames:
        path_parts = fname.split(os.sep)
        
        # LOM name (robust parsing)
        if len(path_parts) >= 2:
            # Attempt to extract LOM name from parent directory, avoiding file-related parts
            this_lom_candidate = path_parts[-2]
            # Simple heuristic: if it contains 'data' or 'run', it might be a data directory not LOM name.
            # This is a bit crude but tries to guess if the parent folder is a run folder or LOM folder.
            if 'data' not in this_lom_candidate.lower() and 'run' not in this_lom_candidate.lower(): 
                 if lomname is None:
                    lomname = this_lom_candidate
                 elif lomname != this_lom_candidate:
                    warnings.warn(f'LOM name inconsistency found: {this_lom_candidate} {lomname}' )
        
        # Run ID parsing
        file_base = os.path.basename(fname)
        run_parts = file_base.split('_')
        if len(run_parts) >= 3:
            try:
                current_runid = int(run_parts[-2].replace('run',''))
                if runid is None:
                    runid = current_runid
                elif runid != current_runid:
                    warnings.warn(f'Run ID inconsistency found: {current_runid} {runid}' )
            except (ValueError, IndexError):
                pass 
        
        # Populate pmt_to_infile_map
        try:
            ch_str_parts = file_base.split(".")
            pmt_id = -1
            if len(ch_str_parts) >= 2:
                potential_ch_part = ch_str_parts[-2]
                if '_' in potential_ch_part:
                    pmt_id = int(potential_ch_part.split('_')[-1])
                else:
                    pmt_id = int(potential_ch_part)
            else:
                pmt_id = int(file_base.split("_")[-1].split(".")[0])
            
            if 0 <= pmt_id < len(plotting_map): # Ensure PMT ID is within expected range (0-19)
                pmt_to_infile_map[pmt_id] = fname
        except (ValueError, IndexError):
            warnings.warn(f"Could not parse PMT ID from filename: {fname}. Skipping for PMT map.")

    if lomname:
        print(f'This LOM is: {lomname}')
    if runid is not None:
        print(f'Run ID: {runid}')
    # --- End: File Scanning and PMT Map Creation ---


    # --- Start: Plotting All Charge Distributions (using newly defined functions) ---
    print("\n--- Plotting Overall Charge Distributions (pC) ---")
    # For overall view, we need to pass a list of *all* unique PMT files
    # Create a list of only unique HDF files (from pmt_to_infile_map values)
    unique_pmt_files = list(pmt_to_infile_map.values())

    plot_all_charge_dists(unique_pmt_files, [-0.1, 1.5], unit_label='Charge (pC)', save_suffix='overall_pC', output_dir=args.output_dir)

    print("\n--- Plotting Overall Charge Distributions (PE) ---")
    assumed_gain = 5e6 # Example gain
    plot_all_charge_dists_pe(unique_pmt_files, assumed_gain=assumed_gain, save_suffix='overall_pe', output_dir=args.output_dir)
    # --- End: Plotting All Charge Distributions ---


    # --- Start: Coincidence Analysis (combined from muon_coincidence.py and muon_analysis_coincidence_script.py) ---
    print("\n--- Starting Coincidence Analysis ---")

    # Coincidence Parameters (can be made configurable via argparse if desired)
    min_charge_pC_for_coincidence = 0.0 # Min charge (pC) for event to be considered for coincidence
    max_charge_pC_for_coincidence = 200.0 # Max charge (pC) for event to be considered for coincidence
    coincidence_window_ns = 100.0 # Time window for coincidence in ns

    # Store all filtered timestamps and PMT IDs for all events for all PMTs
    all_pmt_event_times = {} # {pmt_id: [t1, t2, ...]}
    all_pmt_livetimes = {} # {pmt_id: livetime_sec}
    
    # Populate all_pmt_event_times and calculate livetimes for all PMTs
    print("Calculating Livetimes and Filtering Events for Coincidence...")
    pmt_ids_with_data = sorted(pmt_to_infile_map.keys()) # Only iterate PMTs for which we have files
    
    for pmt_id in pmt_ids_with_data:
        infile_pmt = pmt_to_infile_map[pmt_id]
        
        # Get all FPGA times for this PMT (needed for livetime calculation)
        data_all_fpga = read_hdffile(infile_pmt, quiet=True)
        all_raw_fpga_times = np.array(data_all_fpga["data"]["FPGAtime"][()])
        
        # Calculate livetime for this PMT (simple calculation based on first/last timestamp)
        pmt_livetime = 0.0
        if all_raw_fpga_times.size > 1:
            start_t = all_raw_fpga_times[0]
            end_t = all_raw_fpga_times[-1]
            pmt_livetime = (end_t - start_t) / 1e9 # ns -> s
            if pmt_livetime <= 0: pmt_livetime = 0.0 # Handle non-positive livetime
        
        all_pmt_livetimes[pmt_id] = pmt_livetime
        
        # Get timestamps of events passing the charge threshold range for coincidence
        filtered_timestamps, _ = get_timestamps(infile_pmt, min_charge_pC=min_charge_pC_for_coincidence, max_charge_pC=max_charge_pC_for_coincidence)
        all_pmt_event_times[pmt_id] = np.sort(np.array(filtered_timestamps)) # Ensure sorted numpy array

        print(f"PMT {pmt_id}: Filtered Events = {len(filtered_timestamps)}, Calculated Livetime = {pmt_livetime:.2f}s")


    # --- Coincidence Matrix (Heatmap) Generation (from muon_coincidence.py) ---
    print("\nBuilding Coincidence Matrix (Heatmap)...")
    max_pmt_id_in_data = max(pmt_ids_with_data) if pmt_ids_with_data else 0
    
    coincidence_matrix_counts = np.zeros((max_pmt_id_in_data + 1, max_pmt_id_in_data + 1), dtype=int)
    coincidence_matrix_rates = np.zeros((max_pmt_id_in_data + 1, max_pmt_id_in_data + 1), dtype=float)
    
    for i_pmt in pmt_ids_with_data:
        for j_pmt in pmt_ids_with_data:
            if i_pmt == j_pmt: # Skip self-coincidence for pair-wise matrix
                continue

            times_i = all_pmt_event_times[i_pmt]
            times_j = all_pmt_event_times[j_pmt]
            
            # Use event_matching function for pairwise count
            _, count = event_matching(times_i, times_j, window_ns=coincidence_window_ns)
            
            coincidence_matrix_counts[i_pmt, j_pmt] = count
            
            livetime_i = all_pmt_livetimes.get(i_pmt, 0.0)
            livetime_j = all_pmt_livetimes.get(j_pmt, 0.0)
            
            # Use the minimum of the two livetimes for pairwise rate calculation (common approach)
            common_livetime = min(livetime_i, livetime_j) 
            if common_livetime > 0:
                coincidence_matrix_rates[i_pmt, j_pmt] = count / common_livetime
            else:
                coincidence_matrix_rates[i_pmt, j_pmt] = 0.0

    print("Coincidence Counts Matrix:")
    print(coincidence_matrix_counts)
    print("\nCoincidence Rates Matrix (Hz):")
    print(coincidence_matrix_rates)

    fig_matrix = plt.figure(figsize=(10, 8))
    ax_matrix = fig_matrix.add_subplot(111)
    
    pmt_labels_for_matrix = np.arange(max_pmt_id_in_data + 1) # Labels from 0 to max_pmt_id
    mesh = ax_matrix.pcolormesh(pmt_labels_for_matrix, pmt_labels_for_matrix, 
                                 coincidence_matrix_rates, cmap='Blues', edgecolors='k', linewidth=0.5)
    fig_matrix.colorbar(mesh, ax=ax_matrix, label='Coincidence Rate [Hz]')
    
    ax_matrix.set_xticks(pmt_labels_for_matrix + 0.5)
    ax_matrix.set_yticks(pmt_labels_for_matrix + 0.5)
    ax_matrix.set_xticklabels(pmt_labels_for_matrix)
    ax_matrix.set_yticklabels(pmt_labels_for_matrix)
    
    ax_matrix.set_xlabel('Channel ID', fontsize=12)
    ax_matrix.set_ylabel('Channel ID', fontsize=12)
    ax_matrix.set_title(f'Coincidence Rate Matrix (Threshold=[{min_charge_pC_for_coincidence},{max_charge_pC_for_coincidence}]pC, Window={coincidence_window_ns}ns)', fontsize=14)
    ax_matrix.invert_yaxis()
    ax_matrix.set_aspect('equal', adjustable='box')
    
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    fig_matrix.savefig(f"{args.output_dir}/muon_coincidence_matrix.pdf")
    plt.close(fig_matrix)
    print(f'Saved {args.output_dir}/muon_coincidence_matrix.pdf')


    # --- Coincidence Rate Plot (PMT vs Total Rate) (from muon_analysis_coincidence_script.py) ---
    print("\nPlotting PMT vs Total Coincidence Rate...")
    total_coincidence_rates_per_pmt = np.sum(coincidence_matrix_rates, axis=1) # Summing across rows
    
    fig_rate_total = plt.figure(figsize=(10, 6))
    ax_rate_total = fig_rate_total.add_subplot(1,1,1)
    # Ensure to plot only for PMTs that actually have data
    pmts_to_plot_rate = [p for p in pmt_ids_with_data if p <= max_pmt_id_in_data] # Filter to existing PMTs within matrix bounds
    ax_rate_total.scatter(pmts_to_plot_rate, total_coincidence_rates_per_pmt[pmts_to_plot_rate], color='blue', s=50)
    ax_rate_total.set_xlabel('PMT Channel ID',fontsize=10)
    ax_rate_total.set_ylabel('Total Coincidence Rate [1/s]',fontsize=10)
    ax_rate_total.set_title(f'Total Coincidence Rate per PMT (Threshold=[{min_charge_pC_for_coincidence},{max_charge_pC_for_coincidence}]pC, Window={coincidence_window_ns}ns)', fontsize=12)
    ax_rate_total.tick_params(labelsize=10)
    ax_rate_total.grid('xy', linestyle=':', lw=0.5)
    plt.tight_layout()
    fig_rate_total.savefig(f"{args.output_dir}/muon_total_coincidence_rate_plot.pdf")    
    plt.close(fig_rate_total)
    print(f'Saved {args.output_dir}/muon_total_coincidence_rate_plot.pdf')


    # --- Waveforms of Coincidence Events (from muon_analysis_coincidence_script.py) ---
    print("\n--- Plotting Coincident Event Waveforms ---")
    num_waveforms_to_plot = 5 # Limit to plotting first 5 events to avoid too many files
    waveforms_plotted_count = 0
    
    output_dir_waveforms = f"{args.output_dir}/event_waveforms"
    if not os.path.exists(output_dir_waveforms):
        os.makedirs(output_dir_waveforms)

    # Re-use anchor_timestamps_sorted from earlier for event loop
    
    # Iterate through anchor events to find those that are coincident with at least one other PMT
    for iev_anchor, anchor_ts in enumerate(anchor_timestamps_sorted):
        if waveforms_plotted_count >= num_waveforms_to_plot:
            break

        is_multi_pmt_coincidence = False
        active_pmts_in_event = [] # List of PMT IDs that are coincident at this anchor_ts
        
        # Check if this anchor event has a coincidence with *any* target PMT
        # Loop through target_pmts_ids (0-15) and use their corresponding match_results_all_targets entry
        for pmt_idx_in_list, pmt_id in enumerate(target_pmts_ids): 
            if iev_anchor < len(match_results_all_targets[pmt_idx_in_list]) and \
               match_results_all_targets[pmt_idx_in_list][iev_anchor][1] is not None:
                is_multi_pmt_coincidence = True
                active_pmts_in_event.append(pmt_id)

        if not is_multi_pmt_coincidence:
            continue

        print('----------------------------------')
        print(f"Coincident Event {waveforms_plotted_count + 1} at Anchor Time: {anchor_ts} ns (PMTs: {active_pmts_in_event})")
        
        fig_all_wf = plt.figure(figsize=(15, 12)) 
        
        plotted_pmts_in_this_event_fig = set()

        for pmt_id in target_pmts_ids: # Iterate through all PMT IDs (0-15)
            if pmt_id in plotted_pmts_in_this_event_fig:
                continue

            subplot_idx = plotting_map[pmt_id]
            ax_wf = fig_all_wf.add_subplot(4,5, subplot_idx)
            
            pmt_infile = pmt_to_infile_map.get(pmt_id)
            if pmt_infile is None:
                ax_wf.text(0.5, 0.5, 'File N/A', transform=ax_wf.transAxes, fontsize=10, color='gray', ha='center', va='center')
                ax_wf.set_title(f"pmt{pmt_id}", fontsize=8) 
                ax_wf.tick_params(labelsize=6)
                continue

            coincident_ts_for_pmt = None
            target_pmt_idx_in_list = target_pmts_ids.index(pmt_id)
            if iev_anchor < len(match_results_all_targets[target_pmt_idx_in_list]) and \
               match_results_all_targets[target_pmt_idx_in_list][iev_anchor][1] is not None:
                coincident_ts_for_pmt = match_results_all_targets[target_pmt_idx_in_list][iev_anchor][1]

            if coincident_ts_for_pmt is not None:
                x, wf1, wf2 = get_waveform_at_this_timestamp(pmt_infile, coincident_ts_for_pmt)
                if len(wf1) > 0:
                    ax_wf.plot(x, wf1, lw=0.5, label=f"pmt{pmt_id}/\n@{coincident_ts_for_pmt}")
                else: 
                    ax_wf.text(0.5, 0.5, 'No WF Data', transform=ax_wf.transAxes, fontsize=10, color='red', ha='center', va='center')
            else: 
                ax_wf.text(0.5, 0.5, 'No Coincident Hit', transform=ax_wf.transAxes, fontsize=10, color='gray', ha='center', va='center')

            ax_wf.grid('xy', linestyle=':', lw=0.5)
            ax_wf.set_xlim([100,800]) 
            ax_wf.set_ylim([0,4096])
            ax_wf.tick_params(labelsize=6)
            
            if subplot_idx in [5, 17,18,19,20]:
                ax_wf.tick_params(axis='y', length=0)
                ax_wf.set_yticklabels([])
                ax_wf.set_xlabel('Time (ns)',fontsize=6)
            elif subplot_idx in [ 1,6,11]:
                ax_wf.tick_params(axis='x', length=0)
                ax_wf.set_xticklabels([])
                ax_wf.set_ylabel('ch1 ADC count',fontsize=6)
            elif subplot_idx==16:
                ax_wf.set_xlabel('Time (ns)',fontsize=6)
                ax_wf.set_ylabel('ch1 ADC count',fontsize=6)
            else:
                ax_wf.tick_params(axis='both', length=0)
                ax_wf.set_yticklabels([])
                ax_wf.set_xticklabels([])
            ax_wf.legend(fontsize=6, loc='upper right')
            plotted_pmts_in_this_event_fig.add(pmt_id)

        fig_all_wf.subplots_adjust(wspace=0, hspace=0)
        fig_all_wf.savefig(f"{output_dir_waveforms}/event_{iev_anchor:03d}.pdf")
        plt.close(fig_all_wf)
        waveforms_plotted_count += 1
        print('----------------------------------')

    if waveforms_plotted_count == 0:
        print("No multi-PMT coincident events found for waveform plotting with current criteria.")
    # --- End: Waveforms of Coincidence Events ---


    # --- Start: Aggregated Charge Distributions (Overall debug_muon_dists.pdf) ---
    # This plot visualizes the aggregated charge distribution from all files combined for each PMT.
    print("\n--- Plotting Aggregated Charge Distributions (pC) ---")
    fig_charge_agg = plt.figure(figsize=(15, 12)) 
    for pmt in range(len(plotting_map)): 
        if pmt >= len(all_charge_hists) or not all_charge_hists[pmt]: 
            continue

        # Proper aggregation of y_sum across multiple files for the same PMT
        if len(all_charge_hists[pmt]) > 0: # Ensure there's at least one data set
            x_sum_combined = all_charge_hists[pmt][0][0] # Take x-bins from the first loaded file
            y_sum_combined = np.zeros_like(x_sum_combined, dtype=float)
            for x_data, y_data in all_charge_hists[pmt]:
                # Assuming x-bins are consistent across files for the same PMT
                y_sum_combined += np.array(y_data)
            x_sum = x_sum_combined
            y_sum = y_sum_combined
        else:
            continue # Should not happen due to outer check, but defensive

        subplot_idx_charge = plotting_map[pmt] 
        if subplot_idx_charge < 1 or subplot_idx_charge > 20: 
            continue

        ax_charge = fig_charge_agg.add_subplot(4,5, subplot_idx_charge)
        ax_charge.plot(x_sum, y_sum, lw=0.5, label=f"pmt{pmt}")            
        ax_charge.grid('xy', linestyle=':', lw=0.5)
        ax_charge.set_xlim([-0.1,200]) # Consistent with wide range
        ax_charge.set_yscale('log')
        ax_charge.tick_params(labelsize=6)
                
        if subplot_idx_charge in [5, 17,18,19,20]:
            ax_charge.tick_params(axis='y', length=0)
            ax_charge.set_yticklabels([])
            ax_charge.set_xlabel('Charge (pC)',fontsize=6)
        elif subplot_idx_charge in [ 1,6,11]:
            ax_charge.tick_params(axis='x', length=0)
            ax_charge.set_xticklabels([])
        elif subplot_idx_charge==16:
            ax_charge.set_xlabel('Charge (pC)',fontsize=6)
        else:
            ax_charge.tick_params(axis='both', length=0)
            ax_charge.set_yticklabels([])            
            ax_charge.set_xticklabels([])

        ax_charge.legend(fontsize=6, loc='upper right')

    fig_charge_agg.subplots_adjust(wspace=0, hspace=0)
    
    output_dir_debug_dists = args.output_dir # Use overall output dir
    if not os.path.exists(output_dir_debug_dists):
        os.makedirs(output_dir_debug_dists)

    fig_charge_agg.savefig(f"{output_dir_debug_dists}/debug_muon_dists.pdf")    
    plt.close(fig_charge_agg)
    print(f'Saved {output_dir_debug_dists}/debug_muon_dists.pdf')
    # --- End: Aggregated Charge Distributions ---

    plt.show() # Keep this if you want plots to show up immediately