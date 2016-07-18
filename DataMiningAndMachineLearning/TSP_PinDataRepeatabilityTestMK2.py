from Pillow import *
import cv2
import numpy as np
import math
import time
import os
import copy
import sys
from matplotlib import pyplot as plt
from operator import itemgetter
from scipy.interpolate import interp1d
from scipy import ndimage
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
np.set_printoptions(precision=3, suppress=True)
gnb = GaussianNB()

# ('Pin', 'OriginalXcoord', 'OriginalYcoord', 'OriginalPinSize', 'NewXcoord', 'NewYcoord', 'NewPinSize', 'State', 'DifferenceX', 'DifferenceY', 'Displacement', 'Bearing', 'DifferencePinSize', 'DataSet', 'PastImage', 'PresentImage', 'Rx', 'Ry', 'Rz')

DIR           = os.path.join("TSP_Pictures", "RepeatabilityTestMK2", "390.0mm")
data          = np.load(os.path.join(DIR, "dataline110.npy"))
names         = data.dtype.names
Column1       = names.index('DifferenceX')
Column2       = names.index('DifferenceY')
rw            = rw()
saving        = 0
Single        = 0
stat          = 0
statfuncs     = [np.mean, np.var]
statfuncs     = statfuncs[1]
Sets          = 10
SaveDataArray = []
for set_ in range(10, Sets+1): # for set_ in range(2, Sets+1):
    SaveDataLine  = []
    for step in range(1, 2): # for step in range(1, Sets+1):
        train, test, labels1, labels2, label1, label2 = [], [], [], [], [], []
        for values in data[['Displacement', 'DifferenceX', 'DifferenceY', 'DifferencePinSize', 'X', 'Y', 'Z', 'Rx', 'Ry', 'Rz', 'DataSet', 'State', 'Type']]:
            if values['Type'][0] == 'Z':
                if values['State'].any():
                    if (round((390 - abs(values['Z'][0]))*10, 0)) % step == 0:
                        if 0 < values['DataSet'][0] < set_:
                            if Single:
                                if stat:
                                    train.append(statfuncs(np.concatenate(((values[names[Column1]]*10).astype(int), ))))
                                else:
                                    train.append(np.concatenate(((values[names[Column1]]*10).astype(int), )))
                            else:
                                if stat:
                                    train.append((statfuncs((values[names[Column1]]*10).astype(int)), statfuncs((values[names[Column2]]*10).astype(int))))
                                else:
                                    train.append(np.concatenate(((values[names[Column1]]*10).astype(int), (values[names[Column2]]*10).astype(int))))
                            labels1.append(values['Type'][0])
                            labels2.append(str(int(round((390 - abs(values['Z'][0]))*10, 0))))
                        else:
                            if Single:
                                if stat:
                                    test.append(statfuncs(np.concatenate(((values[names[Column1]]*10).astype(int), ))))
                                else:
                                    test.append(np.concatenate(((values[names[Column1]]*10).astype(int), )))
                            else:
                                if stat:
                                    test.append((statfuncs((values[names[Column1]]*10).astype(int)), statfuncs((values[names[Column2]]*10).astype(int))))
                                else:
                                    test.append(np.concatenate(((values[names[Column1]]*10).astype(int), (values[names[Column2]]*10).astype(int))))
                            label1.append(values['Type'][0])
                            label2.append(str(int(round((390 - abs(values['Z'][0]))*10, 0))))
        if Single and stat:
            train   = np.array(train).reshape(-1,1)
            test    = np.array(test).reshape(-1,1)
        else:
            train   = np.array(train)
            test    = np.array(test)
        labels1 = np.array(labels1)
        label1  = np.array(label1)
        labels2 = np.array(labels2)
        label2  = np.array(label2)

        # print "Predicting the Movement Type"
        y_pred1  = gnb.fit(train, labels1).predict(test)
        y_pred1z = zip(y_pred1, label1)
        # print("Number of mislabeled points out of a total %d points : %d" % (test.shape[0], np.array(label1 != y_pred1).sum()))
        # print "Score = {0}% Success".format(gnb.score(test,label1)*100)

        # print "\nPredicting the Depth"
        y_pred2  = gnb.fit(train, labels2).predict(test)
        y_pred2z = zip(y_pred2, label2)
        # print("Number of mislabeled points out of a total %d points : %d" % (test.shape[0], np.array(label2 != y_pred2).sum()))
        # print "Score = {0}%, {1}mm, #{2}".format(round(gnb.score(test,label2)*100, 3), step, set_-1)
        SaveDataLine.append(gnb.score(test,label2))
        if step == 1: break
        print set_-1, step
    SaveDataArray.append(SaveDataLine)
if Single:
    if stat:
        Name = "TSP_PinDataRepeatabilityTestMK2_{0}_{1}.txt".format(str(statfuncs).split(' ')[1], names[Column1])
    else:
        Name = "TSP_PinDataRepeatabilityTestMK2_{0}.txt".format(names[Column1])
else:
    if stat:
        Name = "TSP_PinDataRepeatabilityTestMK2_{0}_{1}_{2}.txt".format(str(statfuncs).split(' ')[1], names[Column1], names[Column2])
    else:
        Name = "TSP_PinDataRepeatabilityTestMK2_{0}_{1}.txt".format(names[Column1], names[Column2])
print Name

if saving:
    rw.writeList2File(Name, SaveDataArray)

fig = plt.figure()
ax = plt.subplot(1,1,1)
plt.title(Name + '\nTraining Sets: {0}    Testing Sets: {1}    Step distance: {2}mm'.format(set_-1, Sets-(set_-1), float(step)/10))
plt.xlabel("Actual distance, label (mm)")
plt.ylabel("Predicted distance, label (mm)")
x1 = [int(i[1])/10 for i in y_pred2z]
y1 = [int(i[0])/10 for i in y_pred2z]
labels2 = [float(i)/10 for i in label2]
best_fit = plt.plot(labels2, labels2, 'r-', label="Correct Classification")
Classifier_Output = plt.scatter(x1, y1, c='blue', marker="x", label="Classifier Output")
plt.annotate('Rotation Y', xy=(-5, -4), xytext=(-10, 0), arrowprops=dict(facecolor='black', shrink=0.2))
plt.annotate('Rotation X', xy=(5, 6), xytext=(0, 10), arrowprops=dict(facecolor='black', shrink=0.2))
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, labels, loc=4)
plt.grid()
plt.show()
