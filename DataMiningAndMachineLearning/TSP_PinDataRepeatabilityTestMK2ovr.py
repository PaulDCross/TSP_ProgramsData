from Pillow import *
import cv2
import numpy as np
import math
import time
import os
import copy
import sys
from operator import itemgetter
from scipy.interpolate import interp1d
from scipy import ndimage
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import datasets

gnb = GaussianNB()
ovr = OneVsRestClassifier(LinearSVC(random_state=0))
# ('Pin', 'OriginalXcoord', 'OriginalYcoord', 'OriginalPinSize', 'NewXcoord', 'NewYcoord', 'NewPinSize', 'State', 'DifferenceX', 'DifferenceY', 'Displacement', 'Bearing', 'DifferencePinSize', 'DataSet', 'PastImage', 'PresentImage', 'Rx', 'Ry', 'Rz')

DIR  = os.path.join("TSP_Pictures", "RepeatabilityTestMK2", "390.0mm")
data = np.load(os.path.join(DIR, "dataline15.npy"))
train, test, labels1, labels2, label1, label2 = [], [], [], [], [], []
for values in data[['Displacement', 'DifferenceX', 'DifferenceY', 'DifferencePinSize', 'X', 'Y', 'Z', 'Rx', 'Ry', 'Rz', 'DataSet', 'State', 'Type']]:
    if values['Type'][0] == 'Z':
        if values['State'].any():
            if (round((390 - abs(values['Z'][0]))*10, 0)) % 6 == 0:
                if values['DataSet'][0] < 4:
                    train.append(np.concatenate(((values['Displacement']*10).astype(int), )))#, (values['DifferenceY']*10).astype(int))))#, (values['DifferencePinSize']*10).astype(int))))
                    labels1.append(values['Type'][0])
                    labels2.append(str(int(round((390 - abs(values['Z'][0]))*10, 0))))
                else:
                    test.append(np.concatenate(((values['Displacement']*10).astype(int), )))#, (values['DifferenceY']*10).astype(int))))#, (values['DifferencePinSize']*10).astype(int))))
                    label1.append(values['Type'][0])
                    label2.append(str(int(round((390 - abs(values['Z'][0]))*10, 0))))

train   = np.array(train)
test    = np.array(test)
labels1 = np.array(labels1)
label1  = np.array(label1)
labels2 = np.array(labels2)
label2  = np.array(label2)

print "Predicting the Movement Type"
y_pred1  = gnb.fit(train, labels1).predict(test)
y_pred1z = zip(y_pred1, label1)
print("Number of mislabeled points out of a total %d points : %d" % (test.shape[0], np.array(label1 != y_pred1).sum()))
print "Score = {0}% Success".format(gnb.score(test,label1)*100)

print "\nPredicting the Depth"
y_pred2  = ovr.fit(train, labels2).predict(test)
y_pred2z = zip(y_pred2, label2)
print("Number of mislabeled points out of a total %d points : %d" % (test.shape[0], np.array(label2 != y_pred2).sum()))
print "Score = {0}% Success".format(ovr.score(test,label2)*100)
