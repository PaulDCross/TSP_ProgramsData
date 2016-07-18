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
# iris = datasets.load_iris()
gnb = GaussianNB()
# print iris.data, iris.target
# ('Pin', 'OriginalXcoord', 'OriginalYcoord', 'OriginalPinSize', 'NewXcoord', 'NewYcoord', 'NewPinSize', 'State', 'DifferenceX', 'DifferenceY', 'Displacement', 'Bearing', 'DifferencePinSize', 'DataSet', 'PastImage', 'PresentImage', 'Height')

DIR  = os.path.join("TSP_Pictures", "RepeatabilityTestMK1")
# DIR  = os.path.join("TSP_Pictures", "RotationTest", "380.0mm")
data = np.load(os.path.join(DIR, "dataline.npy"))
dx,d,h1,h2 = [],[],[],[]
for values in data[['Displacement','Height', 'DataSet', 'State']]:
    if (round((values['Height'][0]*10), 0)) % 5 == 0:
        if values['State'].any():
            if values['DataSet'][0] < 2:
                dx.append((values['Displacement']*10).astype(int))
                h1.append(int(round(values['Height'][0]*10, 0)))
            else:
                d.append((values['Displacement']*10).astype(int))
                h2.append(int(round(values['Height'][0]*10, 0)))
train = np.array(dx)
test = np.array(d)
labels = np.array(h1)
label = np.array(h2)
y_pred = gnb.fit(train, labels).predict(test)
print zip(y_pred, label)
print("Number of mislabeled points out of a total %d points : %d" % (test.shape[0], np.array(label != y_pred).sum()))
# print y_pred