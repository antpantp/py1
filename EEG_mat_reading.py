# reading mat-files with EEG and ECG
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import itertools

def EEG_read(fpath, fname):
    q = sio.loadmat(fpath + '/' + fname)
    dd = q.get("d")  # reading structure d from the file
    d = {}  # empty dictionary
    d['Fs'] = dd[0][0]["Fs"][0][0]  # sampling frequency
    d['data'] = dd[0][0]["data"].T  # raw signal data, channels are rowwise
    if dd[0][0]["seizureStart"].any():
        d['seizureStart'] = dd[0][0]["seizureStart"].T[0]  # end of seizures, sec
        d['seizureEnd'] = dd[0][0]["seizureEnd"].T[0]  # end of seizures, sec
    d['filt'] = dd[0][0]["filter"][0][0][0][0]  # list with filter settings per channels
    d['labels'] = dd[0][0]["labels"][0]  # array with channel labels
    d['N'] = dd[0][0]["N"][0][0]  # number of samples
    #d['RR'] = dd[0][0]["RR"].T  # RR intervals, interpolated with Fs
    d['RR'] = np.array(list(itertools.chain.from_iterable(dd[0][0]["RR"].T)))
    d['time'] = dd[0][0]["time"][0]  # time base for EEG and ECG channels
    d['RR_raw'] = dd[0][0]["RR_raw"].T[0]  # values of RR intervals, raw
    d['RR_pos'] = dd[0][0]["RR_pos"].T[0]  # position of the RR intervals
    d['RR_2Hz'] = dd[0][0]["RR_2Hz"][0]  # RR interval values, interpolated Fs=2 Hz
    d['time_2Hz'] = dd[0][0]["time_2Hz"][0]  # time base, Fs=2 Hz
    return d, dd

def seizureStartEndInd(d):
    if any(d['seizureStart']):  # if there is a seizure
        Ind = np.array([])  # empty array for indicies
        for seiz in d['seizureStart']:
            I = np.where(d['time'] >= seiz)[0][0]
            Ind = np.append(Ind, I)
        d["seizureStartInd"] = Ind

        Ind = np.array([])  # empty array for indicies
        for seiz in d['seizureEnd']:
            I = np.where(d['time'] >= seiz)[0][0]
            Ind = np.append(Ind, I)
        d["seizureEndInd"] = Ind
    else:
        d['seizureStartInd'] = []
        d['seizureEndInd'] = []
    return d

def EEG_part(d, Start, Stop, channel, flag1, flag2):
    # d -- dictionary with signal data
    # start -- starting point, samples or seconds
    # stop -- end point, samples or seconds
    # channel -- channel number in data
    # flag1 -- string indicating the units for Start and stop, can be either 't' (time), or 's' (samples)
    # fla2 -- integer, if 1, the plot of signal part is plotted
    if flag1 == 't':  # arguments in seconds
        # converting to samples
        Start = int(round(Start * d["Fs"]))
        Stop = int(round(Stop * d["Fs"]))
    else:
        Start = int(Start)
        Stop = int(Stop)
    # extraction of EEG channel part:
    signal = d["data"][channel, Start:Stop]
    time_part = d["time"][Start:Stop]
    if flag2 == 1: # ploting the part
        plt.plot(time_part, signal)
        plt.ylabel(d["labels"][channel][0])
        plt.xlabel("t, sec.")
        plt.show()
    else:
        pass
    signal = np.array(signal)
    time_part = np.array(time_part)
    return signal, time_part

#######################
# function which prepares stacked EEG channel data for classification. It takes data before and after the
# seizure start
def EEG_data_prep_1(qwe, TT, channel, flag1):
    # taking TT sec before and after seizures in different arrays
    # data -- dictionary with data
    # TT -- time, samples or seconds. sec or samples, interval of a signal before and after the seizure start
    # channel -- channel number
    # flag1 -- 's' for samples, 't' for seconds
    during = np.array([])
    before = np.array([])
    for k in range(len(qwe["seizureStart"])):
        sig1, t = EEG_part(qwe, qwe["seizureStart"][k], qwe["seizureStart"][k] + TT, channel, flag1, 0)
        sig2, t = EEG_part(qwe, qwe["seizureStart"][k] - TT, qwe["seizureStart"][k], channel, flag1, 0)
        if k == 0:
            during = sig1
            before = sig2
        else:
            during = np.vstack((during, sig1))
            before = np.vstack((before, sig2))
    during_target = np.ones((during.shape[0], 1))
    before_target = np.zeros((before.shape[0], 1))
    data = np.row_stack((during, before))
    targets = np.ravel(np.row_stack((during_target, before_target)))
    return data, targets
    #########################

def RR_part(d, Start, Stop, RR_type, flag1, flag2):
    # d -- dictionary with signal data
    # start -- starting point, samples or seconds
    # stop -- end point, samples or seconds
    # RR_type -- type of RR data: 'RR' (raw RR intervals, interpolated with Fs), "RR_2Hz" (RR intervals, interpolated with 2 Hz)
    # flag1 -- string indicating the units for Start and stop, can be either 't' (time), or 's' (samples)
    # flag2 -- integer, if 1, the plot of signal part is plotted

    # checking the time input data:
    if flag1 == 't':  # arguments in seconds
        # converting to samples
        Start = int(round(Start * d["Fs"]))
        Stop = int(round(Stop * d["Fs"]))
    else:
        Start = int(Start)
        Stop = int(Stop)

    # extraction of RR signal part:
    if RR_type == 'RR': #  raw RR intervals, interpolated with Fs
        signal = d["RR"][Start:Stop]
        time_part = d["time"][Start:Stop]
    elif RR_type == "RR_2Hz": #  RR intervals, interpolated with 2 Hz
        signal = d["RR_2Hz"][Start:Stop]
        time_part = d["time_2Hz"][Start:Stop]
    else:
        print("Not implemented yet :( ")

    if flag2 == 1: # ploting the part
        plt.plot(time_part, signal)
        plt.ylabel(RR_type)
        plt.xlabel("t, sec.")
        plt.show()
    else:
        pass
    #signal = np.array(signal)
    #time_part = np.array(time_part)
    #signal = list(itertools.chain.from_iterable(signal))
    #time_part = list(itertools.chain.from_iterable(time_part))
    #print('Seizure extracted!')
    return signal, time_part

##################
### data extraction from RR
def RR_data_prep_1(qwe, TT, RR_type, flag1):
    # taking TT sec before and after seizures in different arrays
    # data -- dictionary with data
    # TT -- time, samples or seconds. sec or samples, interval of a signal before and after the seizure start
    # RR_type -- type of RR data: 'RR' (raw RR intervals, interpolated with Fs), "RR_2Hz" (RR intervals, interpolated with 2 Hz)
    # flag1 -- 's' for samples, 't' for seconds
    during = np.array([])
    before = np.array([])
    flag = 'first'
    for k in range(len(qwe["seizureStart"])):
        # checking if there are enough time before and after seizures:
        if qwe["seizureStart"][k] + TT > qwe["time"][-1] or qwe["seizureStart"][k] - TT < 0:
            print("Warning: one seizure is excluded (Time = %.2f).") %TT
            continue
        else:
            sig1, t = RR_part(qwe, qwe["seizureStart"][k], qwe["seizureStart"][k] + TT, RR_type, flag1, 0) # during seizure
            sig2, t = RR_part(qwe, qwe["seizureStart"][k] - TT, qwe["seizureStart"][k], RR_type, flag1, 0) # before seizure
            if flag == 'first':  # working with first applicable seizure
                during = sig1.T
                before = sig2.T
                flag = 'other'
            else:
                during = np.vstack((during, sig1.T))
                before = np.vstack((before, sig2.T))
    data = np.row_stack((during, before))

    during_target = np.ones((during.shape[0], 1))
    before_target = np.zeros((before.shape[0], 1))
    targets = np.ravel(np.row_stack((during_target, before_target)))

    return data, targets
#########################

##################
### data extraction from RR
def RR_data_prep_2(qwe, TT_start, TT_end, RR_type, targ, flag1):
    # taking from TT_start to TT_stop sec before seizures in the array
    # data -- dictionary with data
    # TT_start -- start time of the signal piece, samples or seconds. sec or samples, interval of a signal before the seizure start
    # TT_end -- end time of the signal piece, samples or seconds. sec or samples, interval of a signal before the seizure start
    # targ -- target mark to return with the data
    # flag1 -- 's' for samples, 't' for seconds
    before = np.array([])
    flag = 'first'
    for k in range(len(qwe["seizureStart"])):
        # checking if there are enough time before and after seizures:
        if qwe["seizureStart"][k] + TT_end > qwe["time"][-1] or qwe["seizureStart"][k] - TT_start < 0:
            print("Warning: one seizure is excluded (Time = %.2f).") %TT_start
            continue
        else:
            data, t = RR_part(qwe, qwe["seizureStart"][k] - TT_start, qwe["seizureStart"][k]-TT_end, RR_type, flag1, 0) # before seizure
            if flag == 'first':  # working with first applicable seizure
                before = data.T
                flag = 'other'
            else:
                before = np.vstack((before, data.T))
    data = before #  remove in future updates
    targets = np.ones((before.shape[0], 1))*targ
    #targets = np.ravel(np.row_stack((during_target, before_target)))

    return data, targets
#########################

##################
### data extraction from RR
def RR_data_prep_3(qwe, TT_start, TT_end, RR_type, targ, flag1):
    # WITH ARRAYS
    # taking from TT_start to TT_stop sec before seizures in the array
    # data -- dictionary with data
    # TT_start -- start time of the signal piece, samples or seconds. sec or samples, interval of a signal before the seizure start
    # TT_end -- end time of the signal piece, samples or seconds. sec or samples, interval of a signal before the seizure start
    # targ -- target mark to return with the data
    # flag1 -- 's' for samples, 't' for seconds
    result_data = np.array([])  # empty array for the seizure data
    result_target = np.array([])  #
    if "seizureStart" in qwe.keys():
        flag = 'first'
        if len(qwe["seizureStart"])>0:
            for k in range(len(qwe["seizureStart"])):
                # checking if there are enough time before and after seizures:
                if qwe["seizureStart"][k] + TT_end > qwe["time"][-1] or qwe["seizureStart"][k] - TT_start < 0:
                    #print("Warning: one seizure is excluded (Time = {} sec.)".format(TT_start))
                    continue
                else:
                    data, t = RR_part(qwe, qwe["seizureStart"][k] - TT_start, qwe["seizureStart"][k]-TT_end, RR_type, flag1, 0) # before seizure
                    if flag == 'first':  # working with first applicable seizure
                        result_data = data.T
                        result_target = targ
                        flag = 'other'
                    else:
                        result_data = np.vstack((result_data, data.T))
                        result_target = np.vstack((result_target, targ))
    return result_data, result_target
#########################