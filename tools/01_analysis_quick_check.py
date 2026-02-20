# analysis_quick_check.py (Revised for PE Conversion)
#
# - read a hdf file
# - plot essential information
# - Plot Charge Distributions in pC and Photoelectrons (PE)
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
import sys, os

# === START: Path and Import Management Section ===
# Get the path of the current script (analysis_quick_check.py)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Create path to the 'util' folder
util_path = os.path.join(script_dir, "util")
# Add the 'util' folder path to sys.path if not already present
if util_path not in sys.path:
    sys.path.append(util_path)

# Import necessary modules from the util folder
from HDFWriterModuleInspection import load_dict
from eventHist import * # Ensure eventHist.py is in the util/ folder
from plotting_functions import * # Ensure plotting_functions.py is in the util/ folder

# Check if read_hdffile is defined (should come from plotting_functions.py)
try:
    read_hdffile
except NameError:
    read_hdffile = load_dict
# === END: Path and Import Management Section ===


plotting_map=[6,11,7,12,8,13,9,14,1,16,2,17,3,18,4,19,5,20]

import argparse


def plot_all_charge_dists(filenames, xlims, unit_label='Charge (pC)', save_suffix='ch1'):
    """
    Plots charge distributions for all PMTs in pC.
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
            # Modified to correctly parse channel number for data-spe-run905-0000.00.hdf
            ch_str_parts = base_filename.split(".")
            if len(ch_str_parts) >= 2:
                # Assuming format like 'filename.XX.hdf' or 'filename_XX.hdf'
                # Try to get the part before '.hdf' first, then try to get the number from it
                potential_ch_part = ch_str_parts[-2]
                if '_' in potential_ch_part:
                    ch = int(potential_ch_part.split('_')[-1])
                else: # For SPE data from previous session, it was XX.hdf, so XX is -2
                    ch = int(potential_ch_part)
            else:
                # Fallback for simpler names if necessary, though it might not apply here
                ch = int(base_filename.split("_")[-1].split(".")[0]) # For data_muon_run909_00.hdf format

        except (ValueError, IndexError):
            print(f"Could not parse channel number from filename: {infile}. Skipping.")
            continue

        # Ensure ch is within plotting_map range
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
        if subplot_index in [5, 17,18,19,20]:
            ax_ch1.tick_params(axis='y', length=0)
            ax_ch1.set_yticklabels([])
            ax_ch1.set_xlabel(unit_label,fontsize=6) # Dynamic label
        elif subplot_index in [ 1,6,11]:
            ax_ch1.tick_params(axis='x', length=0)
            ax_ch1.set_xticklabels([])
        elif subplot_index==16:
            ax_ch1.set_xlabel(unit_label,fontsize=6) # Dynamic label
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
            ax_ch2.set_xlabel(unit_label,fontsize=6) # Dynamic label
        elif subplot_index in [ 1,6,11]:
            ax_ch2.tick_params(axis='x', length=0)
            ax_ch2.set_xticklabels([])
        elif subplot_index==16:
            ax_ch2.set_xlabel(unit_label,fontsize=6) # Dynamic label
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
        if subplot_index in [5, 17,18,19,20]:
            ax_ch1_wide.tick_params(axis='y', length=0)
            ax_ch1_wide.set_yticklabels([])
            ax_ch1_wide.set_xlabel(unit_label,fontsize=6) # Dynamic label
        elif subplot_index in [ 1,6,11]:
            ax_ch1_wide.tick_params(axis='x', length=0)
            ax_ch1_wide.set_xticklabels([])
        elif subplot_index==16:
            ax_ch1_wide.set_xlabel(unit_label,fontsize=6) # Dynamic label
        else:
            ax_ch1_wide.tick_params(axis='both', length=0)
            ax_ch1_wide.set_yticklabels([])
            ax_ch1_wide.set_xticklabels([])
        ax_ch1_wide.legend(fontsize=6, loc='upper right')

    if not os.path.exists('./figs'):
        os.makedirs('./figs')

    fig_ch1.subplots_adjust(wspace=0, hspace=0)
    fig_ch1.savefig(f'./figs/all_charge_{save_suffix}.pdf')
    fig_ch2.subplots_adjust(wspace=0, hspace=0)
    fig_ch2.savefig(f'./figs/all_charge_{save_suffix}_ch2.pdf')
    fig_ch1_wide.subplots_adjust(wspace=0, hspace=0)
    fig_ch1_wide.savefig(f'./figs/all_charge_{save_suffix}_wide.pdf')
    plt.close(fig_ch1)
    plt.close(fig_ch2)
    plt.close(fig_ch1_wide)
    print(f'Saved ./figs/all_charge_{save_suffix}.pdf, ./figs/all_charge_{save_suffix}_ch2.pdf, ./figs/all_charge_{save_suffix}_wide.pdf')


def plot_all_charge_dists_pe(filenames, assumed_gain=5e6):
    """
    Plots charge distributions converted to Photoelectrons (PE) for all PMTs.
    Assumes 1 PE charge = assumed_gain * 1.602e-19 C.
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
            # Modified to correctly parse channel number for data-spe-run905-0000.00.hdf
            ch_str_parts = base_filename.split(".")
            if len(ch_str_parts) >= 2:
                potential_ch_part = ch_str_parts[-2]
                if '_' in potential_ch_part:
                    ch = int(potential_ch_part.split('_')[-1])
                else: # For SPE data from previous session, it was XX.hdf, so XX is -2
                    ch = int(potential_ch_part)
            else:
                ch = int(base_filename.split("_")[-1].split(".")[0]) # For data_muon_run909_00.hdf format

        except (ValueError, IndexError):
            print(f"Could not parse channel number from filename: {infile}. Skipping.")
            continue

        # Ensure ch is within plotting_map range
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

    if not os.path.exists('./figs'):
        os.makedirs('./figs')

    fig_ch1.subplots_adjust(wspace=0, hspace=0)
    fig_ch1.savefig(f'./figs/all_charge_pe.pdf')
    fig_ch2.subplots_adjust(wspace=0, hspace=0)
    fig_ch2.savefig(f'./figs/all_charge_pe_ch2.pdf')
    fig_ch1_wide.subplots_adjust(wspace=0, hspace=0)
    fig_ch1_wide.savefig(f'./figs/all_charge_pe_wide.pdf')
    plt.close(fig_ch1)
    plt.close(fig_ch2)
    plt.close(fig_ch1_wide)
    print(f'Saved ./figs/all_charge_pe.pdf, ./figs/all_charge_pe_ch2.pdf, ./figs/all_charge_pe_wide.pdf')


def plot_all_fpgatime(filenames):

    fig = plt.figure(figsize=(12, 9))

    for infile in filenames:
        if not os.path.isfile(infile):
            print(f"Failed. File not found or path is incorrect: {infile}. Skipping.")
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

        ax = fig.add_subplot(4,5, subplot_index)
        fpga_time = data["data"]["FPGAtime"][()]
        fpga_time = np.array(fpga_time)/1e9 # sec

        if fpga_time.size > 0:
            dur = fpga_time[-1] - fpga_time[0]
            print(f"PMT {ch}: Duration={dur:.2f}s, Start={fpga_time[0]:.2f}s, End={fpga_time[-1]:.2f}s")
            x = [ i for i in range(len(fpga_time))]
            ax.plot(x, fpga_time, label=f"pmt{ch}")

            ax.set_ylim([fpga_time[0], fpga_time[-1]])
        else:
            print(f"No FPGAtime data for pmt{ch} in {infile}")
            ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                            fontsize=10, color='red', ha='center', va='center')

        ax.grid('xy', linestyle=':', lw=0.5)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc='upper right')

    if not os.path.exists('./figs'):
        os.makedirs('./figs')

    fig.subplots_adjust(wspace=0, hspace=0)
    outname=f'./figs/all_fpgatime.pdf'
    fig.savefig(outname)
    print(f'Saved {outname}')
    plt.close(fig)


def plot_all_dt(filenames):

    fig = plt.figure(figsize=(12, 9))
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
        x,y = plot_fpga_time_delta(data, no_plot=True)

        ax = fig.add_subplot(4,5, subplot_index)

        if len(x) > 0 and len(y) > 0:
            ax.plot(x, y, lw=0.5, label=f"pmt{ch}", color='blue')
        else:
            print(f"No dt data for pmt{ch} in {infile}")
            ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                            fontsize=10, color='red', ha='center', va='center')


        ax.grid('xy', linestyle=':', lw=0.5)
        ax.set_xticks([-7,-6,-5,-4,-3,-2,-1,0,1])
        ax.set_xlim([-7.5, 1.5])
        ax.tick_params(labelsize=6)
        if subplot_index in [5, 17,18,19,20]:
            ax.tick_params(axis='y', length=0)
            ax.set_yticklabels([])
            ax.set_xlabel('Log10(dt)',fontsize=6)
        elif subplot_index in [ 1,6,11]:
            ax.tick_params(axis='x', length=0)
            ax.set_xticklabels([])
        elif subplot_index==16:
            ax.set_xlabel('Log10(dt)',fontsize=6)
        else:
            ax.tick_params(axis='both', length=0)
            ax.set_yticklabels([])
            ax.set_xticklabels([])

        ax.legend(fontsize=6, loc='upper right')

    if not os.path.exists('./figs'):
        os.makedirs('./figs')

    fig.subplots_adjust(wspace=0, hspace=0)
    outname=f'./figs/all_dt.pdf'
    fig.savefig(outname)
    print(f'Saved {outname}')
    plt.close(fig)


def plot_all_wf(filenames):

    fig = plt.figure(figsize=(12, 9))
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

        nsamples = data["data"]["nsample"][()]
        adc_ch1 = data["data"]["ADC_ch1"][()]
        adc_ch2 = data["data"]["ADC_ch2"][()]

        ax = fig.add_subplot(4,5, subplot_index)
        ax_ch2 = fig_ch2.add_subplot(4,5, subplot_index)

        if adc_ch1.size > 0:
            for iev, wf in enumerate(adc_ch1):
                if iev > 100: continue
                x = [ i *(1e9/60e6) for i in range(len(wf)) ]
                ax.plot(x[:nsamples[iev]], wf[:nsamples[iev]],
                                lw=0.5,)
        else:
            print(f"No ADC_ch1 data for pmt{ch} in {infile}")
            ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                            fontsize=10, color='red', ha='center', va='center')


        if adc_ch2.size > 0:
            for iev, wf in enumerate(adc_ch2):
                if iev > 100: continue
                x = [ i *(1e9/60e6) for i in range(len(wf)) ]
                ax_ch2.plot(x[:nsamples[iev]], wf[:nsamples[iev]],
                                lw=0.5,)
        else:
            print(f"No ADC_ch2 data for pmt{ch} in {infile}")
            ax_ch2.text(0.5, 0.5, 'No Data', transform=ax_ch2.transAxes,
                                fontsize=10, color='red', ha='center', va='center')


        ax.grid('xy', linestyle=':', lw=0.5)
        ax.tick_params(labelsize=6)
        ax.set_ylim([0,4096])
        ax_ch2.grid('xy', linestyle=':', lw=0.5)
        ax_ch2.tick_params(labelsize=6)

        if subplot_index in [5, 17,18,19,20]:
            ax.tick_params(axis='y', length=0)
            ax.set_yticklabels([])
            ax.set_xlabel('ns',fontsize=6)
            ax_ch2.tick_params(axis='y', length=0)
            ax_ch2.set_yticklabels([])
            ax_ch2.set_xlabel('ns',fontsize=6)
        elif subplot_index in [ 1,6,11]:
            ax.tick_params(axis='x', length=0)
            ax.set_xticklabels([])
            ax_ch2.tick_params(axis='x', length=0)
            ax_ch2.set_xticklabels([])
        elif subplot_index==16:
            ax.set_xlabel('ns',fontsize=6)
            ax_ch2.set_xlabel('ns',fontsize=6)
        else:
            ax.tick_params(axis='both', length=0)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax_ch2.tick_params(axis='both', length=0)
            ax_ch2.set_yticklabels([])
            ax_ch2.set_xticklabels([])
        ax.legend([f'ch{ch}'], loc='upper right', fontsize=6)
        ax_ch2.legend([f'ch{ch}'], loc='upper right', fontsize=6)

    if not os.path.exists('./figs'):
        os.makedirs('./figs')

    fig.subplots_adjust(wspace=0, hspace=0)
    fig_ch2.subplots_adjust(wspace=0, hspace=0)
    outname=f'./figs/all_wf.pdf'
    fig.savefig(outname)
    print(f'Saved {outname}')
    outname=f'./figs/all_wf_ch2.pdf'
    fig_ch2.savefig(outname)
    print(f'Saved {outname}')
    plt.close(fig)
    plt.close(fig_ch2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot essential information from HDF files.")
    parser.add_argument("--infilepath", "-i", help="Path to input HDF file(s) with optional wildcard (e.g., 'data/run_*.hdf')", type=str, required=True)
    args = parser.parse_args()

    # Use glob.glob to find all files matching the pattern
    filenames = glob.glob(f"{args.infilepath}")

    if not filenames:
        print(f"No HDF files found matching the pattern: {args.infilepath}")
        sys.exit(1)

    print(f"Found {len(filenames)} files: {filenames}")

    # Plot Charge Distributions in pC (original request)
    print("\n--- Plotting Charge Distributions (pC) ---")
    plot_all_charge_dists(filenames, [-0.1, 1.5], unit_label='Charge (pC)', save_suffix='ch1')

    # Plot Charge Distributions converted to Photoelectrons (PE)
    print("\n--- Plotting Charge Distributions (PE) with assumed gain 5e6 ---")
    # PE conversion factor (pC/PE)
    # 1 electron_charge = 1.602e-19 C
    # assumed_gain = 5e6 electrons/PE
    # charge_per_pe_C = 5e6 * 1.602e-19 = 8.01e-13 C/PE
    # charge_per_pe_pC = 8.01e-13 * 1e12 = 0.801 pC/PE
    pe_conversion_factor = 0.801 # pC per PE
    plot_all_charge_dists_pe(filenames, assumed_gain=5e6) # The assumed_gain is used inside the function

    # Original other plots (FPGAtime, dt, waveforms)
    # plot_all_fpgatime(filenames)
    # plot_all_dt(filenames)
    # plot_all_wf(filenames)

    plt.show()