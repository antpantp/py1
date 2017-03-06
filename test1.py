# for classification of all files in the folder
# 19.02.2017 -- start

import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from EEG_mat_reading import EEG_read, seizureStartEndInd, EEG_part, EEG_data_prep_1, RR_part, RR_data_prep_3
import sklearn
import timeit

# data_path = 'C:/temp'
#data_path = 'D:/Dropbox/EEG_ECG_DATA/MAT/temp'
#data_path = '/media/antp/DATA/Dropbox/EEG_ECG_DATA/MAT/temp'
data_path = '/media/antp/DATA/Dropbox/EEG_ECG_DATA/MAT/Focal Seizures_processed'
s = os.listdir(data_path)  # list with names of the files in directory
#fn = 'volkogon-15.mat'
#fn = 'dukova-15.mat'
#fn = 'tarasov-18.mat'

# initial settings
TT_start = 60  # sec/ samples, before seizure
TT_end = 0
P = 0.8  # ratio for training set
RR_type = "RR"
#time_windows = range(60, 180, )
time_windows = [60]
cl_res = np.empty_like(time_windows, dtype=float)  # classification results for each time window
k = 0  # counter for the classification results for each particular time window
# iterating over all files in the folder, to collect the set of samples

for TT_start in time_windows:  # for various window durations
    data = [] # empty list for the data
    targets = []  # empty list for the targets
    for filename in s:  # for all files
        if filename.endswith(".mat"):
            # reading the file with data
            qwe = EEG_read(data_path, filename)
            # extration the data from all the files and all seizures
            print(filename)
            targ = 1  # target mark for the data, closer to seizure
            (data1, target1) = RR_data_prep_3(qwe, TT_start, TT_end, RR_type, targ, 't')
            targ = 0  # target mark for the data, more far from the seizure
            (data2, target2) = RR_data_prep_3(qwe, 2 * TT_start, TT_start, RR_type, targ, 't')
            data.append(data1)
            data.append(data2)
            targets.append(target1)
            targets.append(target2)
        else:
            continue



# permutation of the items in the set
#perm = np.random.permutation(data.shape[0])  # preparing the numbers to shuffle
#Data = data[perm, :]  # permuted data
#Targets = targets[perm]  # permuted targets
#train = range(int(round(P * Data.shape[0])))  # numbers of elements in training set
#test = range(int(round(P * Data.shape[0])), data.shape[0])  # numbers of elements in test set

#########
# SVM calling and training:
#from sklearn import svm
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#classifier = svm.SVC(
#    kernel='linear')  # (kernel='linear')# other kernels: (kernel='poly', degree=3), (kernel='rbf')

######
#  fitting classifiers
#start_time = timeit.default_timer()
#classifier.fit(Data[train, :], Targets[train])
#elapsed = timeit.default_timer()
#  print "Classifier is fit, elapsed time = %.2f sec." % (elapsed - start_time)

# predicting the class in test set:
# res = classifier.predict(Data[test, :])  # test
#cl_res[k] = classifier.score(Data[test, :], Targets[test])
#k += 1



#################################3


#plt.plot(time_windows, cl_res)
#plt.ylabel("Score")
#plt.xlabel("Window, sec.")
#plt.show()