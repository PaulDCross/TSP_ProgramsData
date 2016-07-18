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
from sklearn import datasets

gnb = GaussianNB()
# ('Pin', 'OriginalXcoord', 'OriginalYcoord', 'OriginalPinSize', 'NewXcoord', 'NewYcoord', 'NewPinSize', 'State', 'DifferenceX', 'DifferenceY', 'Displacement', 'Bearing', 'DifferencePinSize', 'DataSet', 'PastImage', 'PresentImage', 'Rx', 'Ry', 'Rz')

DIR  = os.path.join("TSP_Pictures", "RotationTest", "380.0mm")
data = np.load(os.path.join(DIR, "dataline.npy"))
train, test, labels1, labels2, label1, label2 = [], [], [], [], [], []

for values in data[['Displacement', 'OriginalXcoord', 'OriginalYcoord', 'DataSet', 'State', 'Type', 'Sign']]:
    if values['State'].any():
        if values['DataSet'][0] < 2:
            train = train + zip((values['Displacement']*10).astype(int), (values['OriginalXcoord']*10).astype(int), (values['OriginalYcoord']*10).astype(int))
            [labels1.append(value) for value in values['Sign']]
            [labels2.append(value) for value in values['Type']]
        else:
            test = test + zip((values['Displacement']*10).astype(int), (values['OriginalXcoord']*10).astype(int), (values['OriginalYcoord']*10).astype(int))
            [label1.append(value) for value in values['Sign']]
            [label2.append(value) for value in values['Type']]

train   = np.array(train)
test    = np.array(test)
labels1 = np.array(labels1)
label1  = np.array(label1)
labels2 = np.array(labels2)
label2  = np.array(label2)

print "Predicting the direction: Positive or Negative"
y_pred1 = gnb.fit(train, labels1).predict(test)
# print zip(y_pred1, label1)
print("Number of mislabeled points out of a total %d points : %d" % (test.shape[0], np.array(label1 != y_pred1).sum()))
print "Score = {0}% Success".format(gnb.score(test,label1)*100)

print "\nPredicting the Movement Type"
y_pred2 = gnb.fit(train, labels2).predict(test)
# print zip(y_pred2, label2)
print("Number of mislabeled points out of a total %d points : %d" % (test.shape[0], np.array(label2 != y_pred2).sum()))
print "Score = {0}% Success".format(gnb.score(test,label2)*100)
