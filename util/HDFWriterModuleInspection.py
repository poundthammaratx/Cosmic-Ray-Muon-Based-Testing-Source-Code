

import h5py 
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit  
import warnings
warnings.filterwarnings("ignore")
import glob 


# format of Optical Module testing (Particulary, Upgrade LOMs)
'''
X = 1 or 2
/data/Nsample            [Nwfm];             uint16  unit clock
/data/FPGAtime           [Nwfm];             uint64  unit clock
/data/FPGAtcword         [Nwfm];             uint64  unit clock
/data/charge_chX         [Nwfm];             float32 unit ADC
/data/peak_chX           [Nwfm];             float32 unit ADC
/data/time_chX           [Nwfm];             float32 unit clock
/data/charge_fit_chX     [Nwfm];             float32 unit ADC
/data/peak_fit_chX       [Nwfm];             float32 unit ADC
/data/time_fit_chX       [Nwfm];             float32 unit clock
/data/pedestal_chX       [Nwfm];             float32 unit ADC
/data/ADC_chX            [Nwfm][Nsamplemax]; uint16  unit ADC

/metadata/Nwfm           ; int32    unitless
/metadata/voltage10      ; float32  vol   (dynode10 voltage)
/metadata/pwmfreq        ; int32    frequency unit Hz
/metadata/temperature    ; float32  temperature (Celsius)
/metadata/conversion_ch1 ; float32  unit pC/ADC
/metadata/conversion_ch2 ; float32  unit pC/ADC
/metadata/date           ; int64    unix time
/metadata/PMTID          ; int32    unitless
/metadata/PMTIDstr       ; string
/metadata/wubaseID       ; int32    unitless
/metadata/MCUID          ; string   unitless
/metadata/DACvalue       ; uint16   unit ADC
/metadata/runtype        ; uint8    unitless
/metadata/creatorname    ; string  
/metadata/fitversion     ; uint8
/metadata/description    ; string
/metadata/userdict       ; dictionary to save extra information

runtype: 1 self trigger mode by ch1
       : 2 external trigger mode by ch2
       : 3 CPU trigger mode

userdict can store additional information as a dictionary, and saved as

/metadata/userdict

in HDF file.

This dictionary can include another dictionary, but
data type must be either number/ndarray/dictionary/string.
To load the saved dictionary, use load_dict() function.


Eg. 

from HDFWriterModuleInspection import load_dict 
f = h5py.File("hdfname.hd5", mode="r")
dict = load_dict(f, "metadata/userdict")



'''


class HDFWriterModuleInspection():

    def __init__(self, Nsample_max, Nwfm_init = 0):
        
        self.PMTID       = None
        self.PMTIDstr    = None
        self.WubaseID    = None
        self.MCUID       = None
        self.CreatorName = None
        self.Description = None
        self.RunType = None
        self.Voltage = None
        self.PWMfreq = None
        self.DACvalue = None
        self.Temperature = None
        self.UserDict = None
        self.Nsample_max = Nsample_max
        self.Nwfm     = 0
        self.Nwfm_box = Nwfm_init
        self.init = True
        self.MakinoFit_20230711(np.linspace(0, 60, 61), 0, 1)
        self.fitversion = 0

        # number of samples to calculate pedestal
        self.t_ped = 15

        # signal peak timing window
        self.t_sig_start = 17
        self.t_sig_stop  = 32


        if(Nwfm_init>0):
            self.data_nsample     = np.empty(self.Nwfm_box, dtype=np.uint16)
            self.data_FPGAtime    = np.zeros(self.Nwfm_box, dtype=np.uint64)
            self.data_FPGAtcword  = np.zeros(self.Nwfm_box, dtype=np.uint64)

            self.data_charge_ch1   = np.zeros(self.Nwfm_box, dtype=np.float32)
            self.data_peak_ch1     = np.zeros(self.Nwfm_box, dtype=np.float32)
            self.data_time_ch1     = np.zeros(self.Nwfm_box, dtype=np.float32)
            self.data_pedestal_ch1 = np.zeros(self.Nwfm_box, dtype=np.float32)
            self.data_ADC_ch1      = np.zeros((self.Nwfm_box, self.Nsample_max), dtype=np.uint16)
            
            self.data_charge_ch2   = np.zeros(self.Nwfm_box, dtype=np.float32)
            self.data_peak_ch2     = np.zeros(self.Nwfm_box, dtype=np.float32)
            self.data_time_ch2     = np.zeros(self.Nwfm_box, dtype=np.float32)
            self.data_pedestal_ch2 = np.zeros(self.Nwfm_box, dtype=np.float32)
            self.data_ADC_ch2      = np.zeros((self.Nwfm_box, self.Nsample_max), dtype=np.uint16)

        else:
            self.data_nsample      = None
            self.data_FPGAtime     = None
            self.data_FPGAtcword   = None
            self.data_charge_ch1   = None
            self.data_peak_ch1     = None
            self.data_time_ch1     = None
            self.data_pedestal_ch1 = None
            self.data_ADC_ch1      = None

            self.data_charge_ch2   = None
            self.data_peak_ch2     = None
            self.data_time_ch2     = None
            self.data_pedestal_ch2 = None
            self.data_ADC_ch2      = None

    

    def fill(self, index, nsample, FPGAtime, FPGAtcword,   
                     charge_ch1, peak_ch1, time_ch1, pedestal_ch1, ADC_ch1,
                     charge_ch2, peak_ch2, time_ch2, pedestal_ch2, ADC_ch2):
        '''
        Fill information of a single waveform into a specific index.
        Users must initialize the length of array in the constructor
        '''
        if(index>self.Nwfm_box-1):
            print("error! index exceeds maximum number of wfm")
            return
        self.data_nsample[index]  = nsample
        self.data_FPGAtime[index] = FPGAtime   
        self.data_FPGAtcword[index] = FPGAtcword
        self.data_charge_ch1[index]   = charge_ch1
        self.data_peak_ch1[index]     = peak_ch1   
        self.data_time_ch1[index]     = time_ch1   
        self.data_pedestal_ch1[index] = pedestal_ch1
        self.data_ADC_ch1[index]      = ADC_ch1

        self.data_charge_ch2[index]   = charge_ch2
        self.data_peak_ch2[index]     = peak_ch2   
        self.data_time_ch2[index]     = time_ch2   
        self.data_pedestal_ch2[index] = pedestal_ch2
        self.data_ADC_ch2[index]      = ADC_ch2


    def fill(self, nsample, FPGAtime, FPGAtcword, 
                   charge_ch1, peak_ch1, time_ch1, pedestal_ch1, ADC_ch1,
                   charge_ch2, peak_ch2, time_ch2, pedestal_ch2, ADC_ch2):
        '''
        Fill information of waveforms as arrays separately for summary tables
        '''

        self.data_nsample     = nsample
        self.data_FPGAtime    = FPGAtime
        self.data_FPGAtcword  = FPGAtcword
        self.data_charge_ch1   = charge_ch1
        self.data_peak_ch1     = peak_ch1
        self.data_time_ch1     = time_ch1
        self.data_pedestal_ch1 = pedestal_ch1
        self.data_ADC_ch1      = ADC_ch1
        self.data_charge_ch2   = charge_ch2
        self.data_peak_ch2     = peak_ch2
        self.data_time_ch2     = time_ch2
        self.data_pedestal_ch2 = pedestal_ch2
        self.data_ADC_ch2      = ADC_ch2

        self.Nwfm = charge_ch1.size
        self.Nwfm_box = charge_ch1.size

    def fill(self, nsample, FPGAtime, FPGAtcword, ADC_ch1, ADC_ch2):
        '''
        Fill ADC waveform information
        ADC[Nwfm][nsamplemax]
        '''
        self.Nwfm = nsample.size
        self.data_nsample     = nsample
        self.data_FPGAtime    = FPGAtime
        self.data_FPGAtcword  = FPGAtcword
        self.data_ADC_ch1     = ADC_ch1
        self.data_ADC_ch2     = ADC_ch2

        # calculate pedestal using the first self.t_ped samples
        self.data_pedestal_ch1 = np.average(ADC_ch1[:,:self.t_ped].astype(np.float32), axis=1)
        self.data_pedestal_ch2 = np.average(ADC_ch2[:,:self.t_ped].astype(np.float32), axis=1)

        # calculate charge by summing samples from self.t_sig_start to self.t_sig_stop
        self.data_charge_ch1 =  np.sum(ADC_ch1[:,self.t_sig_start:self.t_sig_stop].astype(np.float32)-self.data_pedestal_ch1[:,None], axis=1)
        self.data_charge_ch2 = -np.sum(ADC_ch2[:,self.t_sig_start:self.t_sig_stop].astype(np.float32)-self.data_pedestal_ch2[:,None], axis=1)
    
        # calculate peak time by using the maximum (minimum in ch2)
        self.data_time_ch1 = (np.argmax(ADC_ch1[:,self.t_sig_start:self.t_sig_stop], axis=1)+self.t_sig_start).astype(np.float32)
        self.data_time_ch2 = (np.argmin(ADC_ch2[:,self.t_sig_start:self.t_sig_stop], axis=1)+self.t_sig_start).astype(np.float32)
    
        # calculate peak heights (minimum in ch2)
        self.data_peak_ch1 =  np.max(ADC_ch1[:,self.t_sig_start:self.t_sig_stop].astype(np.float32), axis=1)-self.data_pedestal_ch1
        self.data_peak_ch2 = -np.min(ADC_ch2[:,self.t_sig_start:self.t_sig_stop].astype(np.float32), axis=1)+self.data_pedestal_ch2

    def fill_metadata(self, pmtid, pmtidstr, wubaseid, mcuid, creatorname, runtype, \
        voltage10, pwmfreq, dacvalue, temperature, conversion_ch1, conversion_ch2, description="", userdict=None):
        self.PMTID       = np.int32(pmtid)
        self.PMTIDstr    = str(pmtidstr)
        self.WubaseID    = np.int32(wubaseid)
        self.MCUID       = str(mcuid)
        self.CreatorName = str(creatorname)
        self.RunType     = np.uint8(runtype)
        self.Voltage10   = np.float32(voltage10)
        self.PWMfreq     = np.int32(pwmfreq)
        self.DACvalue    = np.uint16(dacvalue)
        self.Temperature = np.float32(temperature)
        self.Description = str(description)
        self.Conversion_ch1 = np.float32(conversion_ch1)
        self.Conversion_ch2 = np.float32(conversion_ch2)
        self.UserDict       = userdict

    def MakinoFit_20230711(self, t, t0, amp):
        tsample = 1000/60.0 # in ns
        b1 = 20.5/tsample
        b2 = 63.5/tsample
        c  = 0.7
        p = -8

        v = amp*( c*np.exp( np.exp( -(t-t0)/b1) ) + np.exp( (t-t0)/b2) )**p
        if(self.init):
            # normalize the template function.
            # tpeak is an offset of the real peak position from t=0
            # vpeak is divided so that amp=1 gives its maximum of 1.
            # norm is an area
            T = np.linspace(-30, 60, 1001)
            vtemp = ( c*np.exp( np.exp( -(T)/b1) ) + np.exp( (T)/b2) )**p
            self.vpeak = np.max(vtemp)
            self.tpeak = T[np.argmax(vtemp)]
            self.norm  = np.sum(vtemp)*(T[1]-T[0])/self.vpeak
            self.init = False
        
        return v/self.vpeak 

    def fit_v0(self, tstart=10, tend=40):
        
        # fit Makino function (MakinoFit_20230711)
        # default waveform fit window is from 10 to 40 samples

        self.data_charge_fit_ch1   = np.empty(self.Nwfm).astype(np.float32)
        self.data_peak_fit_ch1     = np.empty(self.Nwfm).astype(np.float32)
        self.data_time_fit_ch1     = np.empty(self.Nwfm).astype(np.float32)

        self.data_charge_fit_ch2   = np.empty(self.Nwfm).astype(np.float32)
        self.data_peak_fit_ch2     = np.empty(self.Nwfm).astype(np.float32)
        self.data_time_fit_ch2     = np.empty(self.Nwfm).astype(np.float32)

        for i in range(self.Nwfm):
            if(i%100==0):
                print("Fitting: i={0} ({1:.1f}%)".format(i, 100.0*i/self.Nwfm))

            y = self.data_ADC_ch1[i][:self.data_nsample[i]]-self.data_pedestal_ch1[i]
            x = np.arange(self.data_nsample[i])
            
            x = x[tstart:tend]
            y = y[tstart:tend]

            p0_time = np.where(self.data_peak_ch1[i]<4, 17, self.data_time_ch1[i]-self.tpeak)
            p0_peak = np.where(self.data_peak_ch1[i]<4, 0, self.data_peak_ch1[i])

            p0 = [p0_time, p0_peak]
            try:
                popt, pcov = curve_fit(self.MakinoFit_20230711, x, y, p0=p0, maxfev=10000)
                self.data_time_fit_ch1[i]     = popt[0]
                self.data_charge_fit_ch1[i]   = popt[1]*self.norm
                self.data_peak_fit_ch1[i]     = popt[1]
            except:
                print("Fit failed in ch1: eventID=", i)
                self.data_time_fit_ch1[i]     = -1
                self.data_charge_fit_ch1[i]   = -1
                self.data_peak_fit_ch1[i]     = -1
                
            debug = False
            if(debug):
                print("p0=", p0)
                print("popt=", popt)
                print("charge fit = ", self.data_charge_fit_ch1[i])
                print("charge     = ", self.data_charge_ch1[i])
                t = np.linspace(0, 100, 1001)
                plt.plot(x, y, "o")
                plt.plot(t, self.MakinoFit_20230711(t, popt[0], popt[1]))

            y = self.data_ADC_ch2[i][:self.data_nsample[i]]-self.data_pedestal_ch2[i]
            x = np.arange(self.data_nsample[i])
            x = x[tstart:tend]
            y = y[tstart:tend]

            p0 = [self.data_time_ch2[i]-self.tpeak, self.data_peak_ch2[i]]
            try:
                popt, pcov = curve_fit(self.MakinoFit_20230711, x, y, p0=p0, maxfev=10000)
                self.data_time_fit_ch2[i]     = -popt[0]
                self.data_charge_fit_ch2[i]   = -popt[1]*self.norm
                self.data_peak_fit_ch2[i]     = -popt[1]
            except:
                print("Fit failed in ch2: eventID=", i)
                self.data_time_fit_ch2[i]     = -1
                self.data_charge_fit_ch2[i]   = -1
                self.data_peak_fit_ch2[i]     = -1
            if(debug):
                t = np.linspace(0, 100, 1001)
                plt.plot(x, y, "o")
                plt.plot(t, self.MakinoFit_20230711(t, popt[0], popt[1]))
                plt.xlim(tstart, tend)
                plt.show()

    def check_formattype(self):
        OK = True
        if(self.data_nsample.size==0):
            print("Error! size of data_nsample is 0!")
            OK = False
        else:
            if(self.data_nsample[0].dtype.name!="uint16"):
                print("Error! data_nsample is not uint16 but {0}".format(self.data_nsample[0].dtype.name))
                OK = False
            if(self.data_FPGAtime[0].dtype.name!="uint64"):
                print("Error! data_FPGAtime is not uint64 but {0}".format(self.data_FPGAtime[0].dtype.name))
                OK = False    
            if(self.data_FPGAtcword[0].dtype.name!="uint64"):
                print("Error! data_FPGAtcword is not uint64 but {0}".format(self.data_FPGAtcword[0].dtype.name))
                OK = False    
            if(self.data_charge_ch1[0].dtype.name!="float32"):
                print("Error! data_charge_ch1 is not float32 but {0}".format(self.data_charge_ch1[0].dtype.name))
                OK = False    
            if(self.data_time_ch1[0].dtype.name!="float32"):
                print("Error! data_time_ch1 is not float32 but {0}".format(self.data_time_ch1[0].dtype.name))
                OK = False    
            if(self.data_pedestal_ch1[0].dtype.name!="float32"):
                print("Error! data_pedestal_ch1 is not float32 but {0}".format(self.data_pedestal_ch1[0].dtype.name))
                OK = False    
            if(self.data_ADC_ch1[0].dtype.name!="uint16"):
                print("Error! data_ADC_ch1 is not uint16 but {0}".format(self.data_ADC_ch1[0].dtype.name))
                OK = False    
            if(self.data_charge_ch2[0].dtype.name!="float32"):
                print("Error! data_charge_ch2 is not float32 but {0}".format(self.data_charge_ch2[0].dtype.name))
                OK = False    
            if(self.data_time_ch2[0].dtype.name!="float32"):
                print("Error! data_time_ch2 is not float32 but {0}".format(self.data_time_ch2[0].dtype.name))
                OK = False    
            if(self.data_pedestal_ch2[0].dtype.name!="float32"):
                print("Error! data_pedestal_ch2 is not float32 but {0}".format(self.data_pedestal_ch2[0].dtype.name))
                OK = False    
            if(self.data_ADC_ch2[0].dtype.name!="uint16"):
                print("Error! data_ADC_ch2 is not uint16 but {0}".format(self.data_ADC_ch2[0].dtype.name))
                OK = False    

            return OK

    def save_dict(self, h5file, path, dic):

        # argument type checking
        if not isinstance(dic, dict):
            raise ValueError("must provide a dictionary")        

        if not isinstance(path, str):
            raise ValueError("path must be a string")
        if not isinstance(h5file, h5py._hl.files.File):
            raise ValueError("must be an open h5py file")
        # save items to the hdf5 file
        for key, item in dic.items():
            key = str(key)
            if isinstance(item, list):
                item = np.array(item)
                #print(item)
            if not isinstance(key, str):
                raise ValueError("dict keys must be strings to save to hdf5")
            # save strings, numpy.int64, and numpy.float64 types
            if isinstance(item, (np.int64, np.float64, str, np.float64, float, np.float32,int)):
                #print( 'here' )
                h5file[path + key] = item

            # save numpy arrays
            elif isinstance(item, np.ndarray):            
                try:
                    h5file[path + key] = item
                except:
                    item = np.array(item).astype('|S9')
                    h5file[path + key] = item

            # save dictionaries
            elif isinstance(item, dict):
                self.save_dict(h5file, path + key + '/', item)
            # other types cannot be saved and will result in an error
            else:
                #print(item)
                raise ValueError('Cannot save %s type.' % type(item))


    def write(self, filename, compression="gzip"):

        if(self.PMTID is None):
            print("Error! metadata is not yet set!")
            return

        # data format check

        if(not self.check_formattype()):
            print("Error in the format check. Writing will not be performed!")
            return

        f = h5py.File(filename, "w")
        f.create_group("data")
        f.create_group("metadata")
        f["data"].create_dataset("nsample",      data=self.data_nsample, compression=compression)
        f["data"].create_dataset("FPGAtime",     data=self.data_FPGAtime, compression=compression)
        f["data"].create_dataset("FPGAtcword",   data=self.data_FPGAtcword, compression=compression)
        f["data"].create_dataset("charge_ch1",   data=self.data_charge_ch1, compression=compression)
        f["data"].create_dataset("peak_ch1",     data=self.data_peak_ch1, compression=compression)
        f["data"].create_dataset("time_ch1",     data=self.data_time_ch1, compression=compression)
        f["data"].create_dataset("pedestal_ch1", data=self.data_pedestal_ch1, compression=compression)
        f["data"].create_dataset("charge_fit_ch1",   data=self.data_charge_fit_ch1, compression=compression)
        f["data"].create_dataset("peak_fit_ch1",     data=self.data_peak_fit_ch1, compression=compression)
        f["data"].create_dataset("time_fit_ch1",     data=self.data_time_fit_ch1, compression=compression)
        f["data"].create_dataset("ADC_ch1",      data=self.data_ADC_ch1, compression=compression)
        f["data"].create_dataset("charge_ch2",   data=self.data_charge_ch2, compression=compression)
        f["data"].create_dataset("peak_ch2",     data=self.data_peak_ch2, compression=compression)
        f["data"].create_dataset("time_ch2",     data=self.data_time_ch2, compression=compression)
        f["data"].create_dataset("pedestal_ch2", data=self.data_pedestal_ch2, compression=compression)
        f["data"].create_dataset("charge_fit_ch2",   data=self.data_charge_fit_ch2, compression=compression)
        f["data"].create_dataset("peak_fit_ch2",     data=self.data_peak_fit_ch2, compression=compression)
        f["data"].create_dataset("time_fit_ch2",     data=self.data_time_fit_ch2, compression=compression)
        f["data"].create_dataset("ADC_ch2",      data=self.data_ADC_ch2, compression=compression)
        f["metadata"].create_dataset("Nwfm", data=self.Nwfm)
        f["metadata"].create_dataset("Nsample_max", data=self.Nsample_max)
        f["metadata"].create_dataset("conversion_ch1", data=self.Conversion_ch1)
        f["metadata"].create_dataset("conversion_ch2", data=self.Conversion_ch2)
        f["metadata"].create_dataset("PMTID", data=self.PMTID)
        f["metadata"].create_dataset("PMTIDstr", data=self.PMTIDstr, dtype=h5py.special_dtype(vlen=str))
        f["metadata"].create_dataset("wubaseID", data=self.WubaseID) 
        f["metadata"].create_dataset("MCUID",   data=self.MCUID, dtype=h5py.special_dtype(vlen=str)) 
        f["metadata"].create_dataset("voltage10", data=self.Voltage10) 
        f["metadata"].create_dataset("PWMfreq", data=self.PWMfreq) 
        f["metadata"].create_dataset("DACvalue", data=self.DACvalue) 
        f["metadata"].create_dataset("temperature", data=self.Temperature) 
        f["metadata"].create_dataset("runtype", data=self.RunType) 
        f["metadata"].create_dataset("creatorname", data=self.CreatorName, dtype=h5py.special_dtype(vlen=str))
        f["metadata"].create_dataset("description", data=self.Description, dtype=h5py.special_dtype(vlen=str))
        f["metadata"].create_dataset("date", data=time.time())
        f["metadata"].create_dataset("fitversion", data=self.fitversion)
        if(self.UserDict is not None):
            self.save_dict(f, "metadata/userdict/", self.UserDict)
        
        f.close()

def load_dict(h5file, path, encode_binary=True): 
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
            if(encode_binary and isinstance(ans[key], bytes)):
                try:
                    ans[key] = ans[key].decode("utf-8")
                except (UnicodeDecodeError, AttributeError):
                    print("warning: {0} could not be decoded. Save as byte format".format(ans[key]))
                    pass
                    
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = load_dict(h5file, path + '/' + key + '/')
    return ans  


if(__name__ == "__main__"):

    # preparation of metadata information
    Nwfm = 1000
    Nsample_max = 50
    PMTID    = 4024
    PMTIDstr = "BB{0}".format(PMTID)
    wubaseID = -1
    MCUID = "mynameissomething"
    runtype = 0
    voltage10 = 85
    dacvalue = 400
    pwmfreq = 111100
    temperature = 25
    conversion_ch1 = 1.0/100 # temporary
    conversion_ch2 = 1.0/100/40 # temporary
    yourname = "Nobu"
    description = "test"

    # userdict can store additional information as a dictionary, 
    # and saved as
    # /metadata/userdict.
    # This dictionary can include another dictionary, but
    # data type must be either number/ndarray/dictionary/string.
    # To load the saved dictionary, use load_dict() function.
    # Eg. 
    # f = h5py.File("hdfname.hd5", mode="r")
    # dict = load_dict(f, "metadata/userdict")
    userdict = {}
    userdict["setupname"] = "laser intensity is 10%"
    userdict["somevariable"] = np.array([1,2,3])
    userdict["somevariable2"] = np.array([1.1,2.3,3.4])
    subsubdict = {"hello":"is a greeting"}
    userdict["subdict"] = {"name":"A", "date":123, "ohmygoodness":777, "tips":subsubdict}



    # a class to save HDF file
    hdf = HDFWriterModuleInspection(Nsample_max)

    # set the metadata
    hdf.fill_metadata(PMTID, PMTIDstr, wubaseID, MCUID, yourname, runtype, voltage10, \
        pwmfreq, dacvalue, temperature, conversion_ch1, conversion_ch2, description, userdict)


    ## ---------------- create dummy data ------------------##
    ## This part is replaced from real extraction of signal ##
    adcs_ch1 = np.reshape(np.random.normal(0, 2, Nwfm*Nsample_max).astype(np.uint16), (Nwfm, Nsample_max))
    adcs_ch1 += 400

    adcs_ch2 = np.reshape(np.random.normal(0, 2, Nwfm*Nsample_max).astype(np.uint16), (Nwfm, Nsample_max))
    adcs_ch2 += 3800

    
    # number of samples. in this dummy creation, this is fixed, but real data would dynamically change.
    nsample = (np.ones(Nwfm)*Nsample_max).astype(np.uint16)
    
    # FPGA time
    FPGA_time = np.ones(Nwfm).astype(np.uint64)*1000
    FPGA_tcword = np.ones(Nwfm).astype(np.uint64)*1000

    ## ------------- end of dummy data creation ------------------##

    # save the waveform and their basic values
    hdf.fill(nsample, FPGA_time, FPGA_tcword, adcs_ch1, adcs_ch2)
    
    # fit Makino function to precisely evealuate charge/time/peak.
    hdf.fit_v0()

    hdf.write("testdata.hd5")
