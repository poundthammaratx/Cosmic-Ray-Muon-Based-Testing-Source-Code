#
# - read a hdf file
# - plot essential information
# test test
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
from sys import argv
import sys, os
sys.path.append("../util/")
from HDFWriterModuleInspection import load_dict
from HDFWriterModuleInspection import HDFWriterModuleInspection
from eventHist import *
import json

def read_hdffile(filename, quiet=False):
    
    f = h5py.File(filename, mode="r")

    userdict = load_dict(f, "metadata/userdict")
    
    FPGAtime = f["data"]["FPGAtime"][()]
    duration = FPGAtime[-1]-FPGAtime[0] 

    if quiet: return f
    
    print('============================')
    print('PMT: ', f["metadata"]["PMTIDstr"][()].decode("utf-8") )
    print('wuBase: ',  f["metadata"]["wubaseID"][()] )
    print('wuBase MCU: ', f["metadata"]["MCUID"][()].decode("utf-8") )
    print('\n')
    print('Nevents: ',  f["metadata"]["Nwfm"][()] )
    print('Duration: ', round(duration/60e6, 1), 'sec? (this may not be correct)' )
    print('Trigger rate: ', round(  f["metadata"]["Nwfm"][()] /(duration/60e6),1), '/s (this may not be correct)')
    print('Temperature:', f["metadata"]["temperature"][()] )
    print('============================')

    return f

def func_gaus(x, a, b, x0):
    return a * np.exp(- (x-x0)**2/2/b**2) 

def func_exp(x, ea, et):
    return ea*np.exp(-1* (x/et) )

def func_gaus_exp(x, a, b, x0, ea, et):
    return func_gaus(x, a, b, x0) + func_exp(x, ea, et)


def is_overflow(adc_ch1, adc_ch2):

    ch1_overflow=0
    ch2_overflow=0

    for v in adc_ch1:
        if v==4095: ch1_overflow=1

    for v in adc_ch2:
        if v==0: ch2_overflow=1

    return ch1_overflow, ch2_overflow
        

def plot_pedestal(data, outpath='./', no_plot=False,):

    ch1 = data["data"]["pedestal_ch1"][()]   
    ch2 = data["data"]["pedestal_ch2"][()]       
    
    # debug, ch1 adc peak distribution
    h_ch1 = eventHist(0,4096,4096,"sample","Ch1 peak pedestal ADC","Events")
    h_ch2 = eventHist(0,4096,4096,"sample","Ch2 peak pedestal ADC","Events")
    for c in ch1:
        h_ch1.increment(c)
    for c in ch2:
        h_ch2.increment(c)    
    x_ch1 = [ h_ch1.xmin + j*h_ch1.dx for j in range( h_ch1.nbins ) ]
    y_ch1 = [ h_ch1.getCounts(j) for j in range( h_ch1.nbins )]
    x_ch2 = [ h_ch2.xmin + j*h_ch2.dx for j in range( h_ch2.nbins ) ]
    y_ch2 = [ h_ch2.getCounts(j) for j in range( h_ch2.nbins )]

    if no_plot:
        return [x_ch1, y_ch1], [x_ch2, y_ch2]

    fig = plt.figure()
    
    # Add first subplot in a 2x1 grid at position 1
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(x_ch1, y_ch1)
    ax1.set_title('ch1')
    ax1.grid('xy', linestyle=':')

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(x_ch2, y_ch2)
    ax2.set_title('ch2')
    ax2.grid('xy', linestyle=':')
    
    plt.tight_layout()
    plt.savefig(f'{outpath}/debug_pedestal.pdf')

def plot_charge(data, outpath='./', no_plot=False, 
                xmin_ch1=-1, xmax_ch1=3, nbin_ch1=114*4,
                xmin_ch2=-1, xmax_ch2=3, nbin_ch2=200,
                log10dtcut=None):

    q_ch1 = data["data"]["charge_ch1"][()]
    q_ch2 = data["data"]["charge_ch2"][()]
    fpga_time = data["data"]["FPGAtime"][()]
    conversion_ch1 = data["metadata"]["conversion_ch1"][()]
    conversion_ch2 = data["metadata"]["conversion_ch2"][()]
    q_ch1 = [ s * (conversion_ch1* 1e-6 * (1/60e6) * 1e12) for s in q_ch1]
    q_ch2 = [ s * (conversion_ch2* 1e-6 * (1/60e6) * 1e12) for s in q_ch2]

    h_ch1 = eventHist(xmin_ch1,xmax_ch1,nbin_ch1,"sample","Ch1 Charge (pC)","Events")
    h_ch2 = eventHist(xmin_ch2,xmax_ch2,nbin_ch2,"sample","Ch2 Charge (pC)","Events")

    last_time=0
    for iev, (q1,q2) in enumerate(zip(q_ch1,q_ch2)):

        if log10dtcut==None:
            h_ch1.increment(q1)
            h_ch2.increment(q2)                        
            continue

        if iev==0:
            last_time = fpga_time[iev]
            continue
        
        dt = float(fpga_time[iev] - last_time)/1e9  # nsec -> sec
        last_time=fpga_time[iev]        
        if np.log10(dt) > (log10dtcut+0.2) or np.log10(dt) < (log10dtcut-0.2):
            continue
        h_ch1.increment(q1)
        h_ch2.increment(q2) 
    
    x_ch1 = [ h_ch1.xmin + j*h_ch1.dx for j in range( h_ch1.nbins ) ]
    y_ch1 = [ h_ch1.getCounts(j) for j in range( h_ch1.nbins )]
    x_ch2 = [ h_ch2.xmin + j*h_ch2.dx for j in range( h_ch2.nbins ) ]
    y_ch2 = [ h_ch2.getCounts(j) for j in range( h_ch2.nbins )]

    if no_plot:
        return [x_ch1, y_ch1], [x_ch2, y_ch2]

    fig = plt.figure()
    
    # Add first subplot in a 2x1 grid at position 1
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(x_ch1, y_ch1)
    ax1.set_title('ch1')
    ax1.grid('xy', linestyle=':')

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(x_ch2, y_ch2)
    ax2.set_title('ch2')
    ax2.grid('xy', linestyle=':')
    
    plt.tight_layout()
    plt.savefig(f'{outpath}/debug_charge.pdf')
    
def plot_charge_w_dt_cut(data, outpath='./', no_plot=True,
                         dt_th=-5.0,
                         xmin_ch1=-1, xmax_ch1=3, nbin_ch1=114*4,
                         xmin_ch2=-1, xmax_ch2=3, nbin_ch2=200):

    fpga_time = data["data"]["FPGAtime"][()]

    q_ch1 = data["data"]["charge_ch1"][()]
    q_ch2 = data["data"]["charge_ch2"][()]    
    conversion_ch1 = data["metadata"]["conversion_ch1"][()]
    conversion_ch2 = data["metadata"]["conversion_ch2"][()]
    q_ch1 = [ s * (conversion_ch1* 1e-6 * (1/60e6) * 1e12) for s in q_ch1]
    q_ch2 = [ s * (conversion_ch2* 1e-6 * (1/60e6) * 1e12) for s in q_ch2]

    q_ch1_selected=[]
    q_ch2_selected=[]    
    # event cut
    for i in range( 1, len(fpga_time) ): # start from indx 1 to have the "previous" event always
        dt = np.log10( float(fpga_time[i] - fpga_time[i-1])/1e9 )
        if dt < dt_th: # if this event is within dt_th from the previous event
            q_ch1_selected.append(q_ch1[i])
            q_ch2_selected.append(q_ch2[i])
    
    h_ch1 = eventHist(xmin_ch1,xmax_ch1,nbin_ch1,"sample","Ch1 Charge (pC)","Events")
    h_ch2 = eventHist(xmin_ch2,xmax_ch2,nbin_ch2,"sample","Ch2 Charge (pC)","Events")    
    for q in q_ch1_selected:
        h_ch1.increment(q)
    for q in q_ch2_selected:
        h_ch2.increment(q)        
    
    x_ch1 = [ h_ch1.xmin + j*h_ch1.dx for j in range( h_ch1.nbins ) ]
    y_ch1 = [ h_ch1.getCounts(j) for j in range( h_ch1.nbins )]
    x_ch2 = [ h_ch2.xmin + j*h_ch2.dx for j in range( h_ch2.nbins ) ]
    y_ch2 = [ h_ch2.getCounts(j) for j in range( h_ch2.nbins )]

    if no_plot:
        return [x_ch1, y_ch1], [x_ch2, y_ch2]
    
def plot_peak(data, outpath):

    p_ch1 = data["data"]["peak_ch1"][()]
    p_ch2 = data["data"]["peak_ch2"][()]
    p_fit_ch1 = data["data"]["peak_fit_ch1"][()]
    p_fit_ch2 = data["data"]["peak_fit_ch2"][()]        

    
    h_ch1 = eventHist(0,4096, 2048,"sample","Ch1 Waveform Peak","Events")
    h_ch2 = eventHist(0,4096, 2048,"sample","Ch2 Waveform Peak","Events")
    h_fit_ch1 = eventHist(0,4096, 2048,"sample","Ch1 Fit Waveform Peak","Events")
    h_fit_ch2 = eventHist(0,4096, 2048,"sample","Ch2 Fit Waveform Peak","Events")
    
    for p in p_ch1:
        h_ch1.increment(p)
    for p in p_ch2:
        h_ch2.increment(p)
    for p in p_fit_ch1:
        h_fit_ch1.increment(p)
    for p in p_fit_ch2:
        h_fit_ch2.increment(p)                
    
    x_ch1 = [ h_ch1.xmin + j*h_ch1.dx for j in range( h_ch1.nbins ) ]
    y_ch1 = [ h_ch1.getCounts(j) for j in range( h_ch1.nbins )]
    x_fit_ch1 = [ h_fit_ch1.xmin + j*h_fit_ch1.dx for j in range( h_fit_ch1.nbins ) ]
    y_fit_ch1 = [ h_fit_ch1.getCounts(j) for j in range( h_fit_ch1.nbins )]    

    x_ch2 = [ h_ch2.xmin + j*h_ch2.dx for j in range( h_ch2.nbins ) ]
    y_ch2 = [ h_ch2.getCounts(j) for j in range( h_ch2.nbins )]
    x_fit_ch2 = [ h_fit_ch2.xmin + j*h_fit_ch2.dx for j in range( h_fit_ch2.nbins ) ]
    y_fit_ch2 = [ h_fit_ch2.getCounts(j) for j in range( h_fit_ch2.nbins )]
    
    fig = plt.figure()
    
    # Add first subplot in a 2x1 grid at position 1
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(x_ch1, y_ch1, color='black')
    ax1.plot(x_fit_ch1, y_fit_ch1, color='red', linestyle=':')    
    ax1.set_title('ch1')
    ax1.grid('xy', linestyle=':')

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(x_ch2, y_ch2, color='black')
    ax2.plot(x_fit_ch2, y_fit_ch2, color='red', linestyle=':')  
    ax2.set_title('ch2')
    ax2.grid('xy', linestyle=':')
    
    plt.tight_layout()
    plt.savefig(f'{outpath}/debug_fit-peak.pdf')

def plot_fit_time(data, outpath):

    time_fit_ch1 = data["data"]["time_fit_ch1"][()]
    time_fit_ch1 = [ t *(1e9/60e6) for t in time_fit_ch1 ]
    time_fit_ch2 = data["data"]["time_fit_ch2"][()]
    time_fit_ch2 = [ t *(1e9/60e6) for t in time_fit_ch2 ]    
     
    h_ch1 = eventHist(0,800, 1600,"sample","Ch1 Fit arrival time","Events")
    h_ch2 = eventHist(0,800, 1600,"sample","Ch2 Fit arrival time","Events")    
    for p in time_fit_ch1:
        h_ch1.increment(p)
    for p in time_fit_ch2:
        h_ch2.increment(p)        
    
    x_ch1 = [ h_ch1.xmin + j*h_ch1.dx for j in range( h_ch1.nbins ) ]
    y_ch1 = [ h_ch1.getCounts(j) for j in range( h_ch1.nbins )]
    x_ch2 = [ h_ch2.xmin + j*h_ch2.dx for j in range( h_ch2.nbins ) ]
    y_ch2 = [ h_ch2.getCounts(j) for j in range( h_ch2.nbins )]

    fig = plt.figure()
    
    # Add first subplot in a 2x1 grid at position 1
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(x_ch1, y_ch1)
    ax1.set_title('ch1')
    ax1.grid('xy', linestyle=':')

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(x_ch2, y_ch2)
    ax2.set_title('ch2')
    ax2.grid('xy', linestyle=':')
    
    plt.tight_layout()
    plt.savefig(f'{outpath}/debug_fit-time.pdf')


def plot_waveforms(data, outpath, nmax=100):

    nsamples = data["data"]["nsample"][()]
    adc_ch1 = data["data"]["ADC_ch1"][()]
    adc_ch2 = data["data"]["ADC_ch2"][()]     
    
    #print(data.t_sig_start)
    #print(data.t_sig_stop)
    #print(data.t_ped)

    fig = plt.figure()
    
    ax1 = fig.add_subplot(2, 1, 1)
    for iev, wf in enumerate(adc_ch1):
        if iev > nmax: continue
        x =  [ i *(1e9/60e6) for i in range(len(wf)) ]        
        ax1.plot(x[:nsamples[iev]], wf[:nsamples[iev]],
                 lw=0.5,)
    ax1.set_title('ch1')
    ax1.grid('xy', linestyle=':')

    ax2 = fig.add_subplot(2, 1, 2)
    for iev, wf in enumerate(adc_ch2):
        if iev > nmax: continue
        x =  [ i *(1e9/60e6) for i in range(len(wf)) ]        
        ax2.plot(x[:nsamples[iev]], wf[:nsamples[iev]],
                 lw=0.5,)
    ax2.set_title('ch2')
    ax2.grid('xy', linestyle=':')
    
    plt.tight_layout()
    plt.savefig(f'{outpath}/debug_waveforms.pdf')

def plot_fpga_time(data, outpath):
    fpga_time = data["data"]["FPGAtime"][()]
    x = [ i for i in range(len(fpga_time))]

    fig = plt.figure()
    ax = fig.add_subplot()        
    ax.plot(x, fpga_time)
    ax.grid('xy', linestyle=':')
    plt.tight_layout()
    plt.savefig(f'{outpath}/debug_fpgatime.pdf')



def plot_fpga_time_delta(data, outpath='./', no_plot=False, xmax= 2, xmin=-8):

    nbins=100

    fpga_time = data["data"]["FPGAtime"][()]
    fpga_time = fpga_time.astype(int)
    delta_t=[]
    for i in range( len(fpga_time)-1 ):
        dt_ = (fpga_time[i+1] - fpga_time[i]) * 1e-9 # sec
        if dt_<0: # should not be the case
            print("Ughhhhhh")
            dt_ = xmin # Fill it to underflow bin for now. 
        else:
            dt_ = np.log10( dt_ )
        delta_t.append(dt_)

    # use "xmax" and "xmin" bins for overflow/under flow entries
    for i in range( len(delta_t) ):
        if delta_t[i] > xmax: 
            delta_t[i]=xmax
        if delta_t[i] < xmin: 
            delta_t[i]=xmin

    h = eventHist(xmin,xmax, nbins,"sample","log10dt","Events")
    for p in delta_t:
        h.increment(p)
    
    x = [ h.xmin + j*h.dx for j in range( h.nbins ) ]
    y = [ h.getCounts(j) for j in range( h.nbins )]

    if no_plot:
        return x, y

    return x, y

def get_livetime(data):

    fpga_time = data["data"]["FPGAtime"][()]
    fpga_time = fpga_time.astype(int)
    start_time=fpga_time[0]
    end_time=fpga_time[-1]
    livetime = end_time - start_time 

    #for i in range( len(fpga_time)-2 ):
    #    if fpga_time[i+1]<fpga_time[i]:
    #        print("Nooooooooo!")
    #        print(fpga_time[i-1], fpga_time[i], fpga_time[i+1],)
        
    # if pulse separation is more than 0.1 sec (100ms), consider the gap as a deadtime.
    # Total livetime will be corrected with accumulated deadtime.
    accumulated_deadtime=0
    last_time=0
    n_corruption=0
    for iev, t in enumerate(fpga_time):
        
        if t<start_time or t>end_time: continue 

        if iev==0: 
            last_time = t
            continue 
        
        dt = t - last_time # nsec. 
        # original type is np.uint64, but calculation goes nuts 
        # when last_time is bigger than t (which can happen because 
        # of corrupted data)

        #print(iev, dt)
        #if dt < 0: # if the previous event is more future than this event
        #    print('Time machine situation here. (t-last_time)<0!', t, last_time,)
        #    #print(fpga_time[iev-2], fpga_time[iev-1], fpga_time[iev],)
        #    continue 

        if dt > 1e11: # more than 100 sec, timestamp gotta be corrupted
            n_corruption+=1
            #print(iev, fpga_time[iev-1]/1e9, fpga_time[iev]/1e9, fpga_time[iev+1]/1e9,# 
            #      last_time/1e9, np.log10(dt))
            continue

                
        #if dt > 1e8: # 0.1 sec        
        if dt > 1e7: # 0.01 sec
            accumulated_deadtime += dt
            #print('check', fpga_time[iev-1]/1e9, fpga_time[iev]/1e9, fpga_time[iev+1]/1e9)

        last_time = t

    corr_livetime= (livetime - accumulated_deadtime)/1e9
    #print(start_time, end_time, livetime, accumulated_deadtime, corr_livetime)
    
    return corr_livetime


def get_trigger_rate(data):

    ntriggered = data['metadata']['Nwfm'][()]
    livetime = get_livetime(data)
    trg_rate = ntriggered / livetime

    return trg_rate, ntriggered, livetime

def get_dark_rate(data, user_deadtime_ns=500):

    # This is different from trigger rate 

    npulses = 0
    livetime = get_livetime(data)
    dark_rate = npulses / livetime

    return dark_rate, npulses, livetime


def get_average_waveform(data, tar_ch=1, qmin=0.7, qmax=0.9):

    # Output containers
    # Each event is sampled ADC data, but with X axis offset based on the fit results.
    # Overlaying many of them will represent a "smooth" raw waveform shape
    res_x = [] 
    res_y = [] 

    # Number of events
    nevent = data["metadata"]["Nwfm"][()]

    # Time stamps for optional dt cut
    fpga_time = data["data"]["FPGAtime"][()]

    # Raw ADC 
    # & Fitted arrival times to shift observed raw data
    # & Charge info for event selection 
    nsamples = data["data"]["nsample"][()]
    if tar_ch==2:
        adc = data["data"]["ADC_ch2"][()]
        time_fit = data["data"]["time_fit_ch2"][()]
        time_fit = [ t *(1e9/60e6) for t in time_fit ] 
        qs = data["data"]["charge_ch2"][()]    
        conversion_ch2 = data["metadata"]["conversion_ch2"][()]    
        qs = [ s * (conversion_ch2* 1e-6 * (1/60e6) * 1e12) for s in qs]
    else:
        adc = data["data"]["ADC_ch1"][()]  
        time_fit = data["data"]["time_fit_ch1"][()]
        time_fit = [ t *(1e9/60e6) for t in time_fit ]
        qs = data["data"]["charge_ch1"][()]
        conversion_ch1 = data["metadata"]["conversion_ch1"][()]
        qs = [ s * (conversion_ch1* 1e-6 * (1/60e6) * 1e12) for s in qs]

    ave_peak_time = np.average(time_fit)

    for iev in range(nevent):

        # charge cut 
        if qs[iev] < qmin or qs[iev] > qmax: continue 
        
        # fit position check 
        #if time_fit[iev] < (ave_peak_time-50.0): continue
        #if time_fit[iev] > (ave_peak_time+50.0): continue

        n = nsamples[iev]
        x =  [ i *(1e9/60e6) for i in range(n) ]  
        x = np.array(x) - time_fit[iev]
        for v in x:
            res_x.append(v)
        for v in adc[iev][:n]:
            res_y.append(v)

    return res_x, res_y



# Function to append data points to a JSON file
def append_data_to_json(file_path, label, x, y):
    # Check if the file exists
    if os.path.exists(file_path):
        # Read the existing data
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
    else:
        # Initialize an empty dictionary if the file does not exist
        data = {}

    # Append the new data points
    if len(x)==0:
        data[label] = {"x": None, "y": None}
    else:
        data[label] = {"x": x, "y": y}

    # Write the updated data back to the JSON file
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def append_result_to_json(file_path, label, x):
    # Check if the file exists
    if os.path.exists(file_path):
        # Read the existing data
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
    else:
        # Initialize an empty dictionary if the file does not exist
        data = {}

    # Append the new data points
    data[label] = {"x": x}

    # Write the updated data back to the JSON file
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
        
def extract_matching_elements(timestamp_ref_ns, timestamp_ns, window_ns=1000.):

    matching_elements=[]
    matching_event_index=[]
    n_fail=0
    for idx_ref, t_ref in enumerate(timestamp_ref_ns):
        found_it = False
        for idx, t in enumerate(timestamp_ns):
            if abs(t_ref-t) <= window_ns:                
                matching_elements.append([t_ref,t])
                matching_event_index.append(idx)
                delta = t_ref-t
                found_it = True
                #print(delta) #
        if found_it == False:
            n_fail+=1

    print(f"Drop ratio: {n_fail} / {len(timestamp_ref_ns)}")
    
    return matching_elements, matching_event_index
