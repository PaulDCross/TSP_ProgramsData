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
rw  = rw()

def makedir(DIR):
    if not os.path.exists(DIR):
        os.makedirs(DIR)
        time.sleep(0.5)

ztool          = 167.5
zcoordinate    = "342.5mm"
directory      = os.path.join("PinDataResults", "RotationTestsTypeSign", "RotationTest{0}".format(ztool))
makedir(directory)
DIR            = os.path.join("TSP_Pictures", "RotationTests", "RotationTest{0}".format(ztool), zcoordinate)
data           = np.load(os.path.join(DIR, "dataline110.npy"))
names          = data.dtype.names
Column0        = names.index('Displacement')
Column1        = names.index('DifferenceX')
Column2        = names.index('DifferenceY')
titles1        = ['Rx', 'Ry']
titles2        = ['P', 'N']
IncludeSign    = 1
All            = 1
differentsigns = 0
savingtext     = 0
Graph          = 1
savingGraph    = 0
show           = 1
Sets           = int(1 + len([name for name in os.listdir(DIR) if os.path.isdir(os.path.join(DIR, name))]))
if IncludeSign:
    print "Including Sign"
    # for title1 in titles1:
        # for title2 in titles2:
    # title1 = titles1[0]
    # title2 = titles2[0]
    if True:
            for single in range(2):
                SaveDataArray  = []
                for set_ in range(10, Sets): # for set_ in range(2, Sets):
                    SaveDataLine  = []
                    for step in range(1, 2): # for step in range(1, Sets):
                        train, test, labels1, labels2, labels3, label1, label2, label3 = [], [], [], [], [], [], [], []
                        for values in data[['Displacement', 'DifferenceX', 'DifferenceY', 'Bearing', 'X', 'Y', 'Z', 'Rx', 'Ry', 'Rz', 'DataSet', 'State', 'Type', 'Sign']]:
                            if All:
                                if values['State'].any():
                                    if (round((data[values['Type'][0]][0][0] - abs(values[values['Type'][0]][0]))*10, 0)) % step == 0:
                                        if 0 < values['DataSet'][0] < set_:
                                            if single:
                                                train.append(((values[names[Column0]]*10).astype(int)))
                                            else:
                                                train.append(np.concatenate(((values[names[Column1]]*10).astype(int), (values[names[Column2]]*10).astype(int))))
                                            labels1.append(values['Type'][0])
                                            labels2.append(values['Sign'][0])
                                            labels3.append(str(int(round((data[values['Type'][0]][0][0] - abs(values[values['Type'][0]][0]))*10, 0)))+values['Type'][0]+values['Sign'][0])
                                        else:
                                            if single:
                                                test.append(((values[names[Column0]]*10).astype(int)))
                                            else:
                                                test.append(np.concatenate(((values[names[Column1]]*10).astype(int), (values[names[Column2]]*10).astype(int))))
                                            label1.append(values['Type'][0])
                                            label2.append(values['Sign'][0])
                                            label3.append(str(int(round((data[values['Type'][0]][0][0] - abs(values[values['Type'][0]][0]))*10, 0)))+values['Type'][0]+values['Sign'][0])
                            else:
                                if differentsigns:
                                    if (values['Type'][0] == title1) and (values['Sign'][0] == title2):
                                        if values['State'].any():
                                            if (round((data[values['Type'][0]][0][0] - abs(values[values['Type'][0]][0]))*10, 0)) % step == 0:
                                                if 0 < values['DataSet'][0] < set_:
                                                    if single:
                                                        train.append(((values[names[Column0]]*10).astype(int)))
                                                    else:
                                                        train.append(np.concatenate(((values[names[Column1]]*10).astype(int), (values[names[Column2]]*10).astype(int))))
                                                    labels1.append(values['Type'][0])
                                                    labels2.append(values['Sign'][0])
                                                    labels3.append(str(int(round((data[values['Type'][0]][0][0] - abs(values[values['Type'][0]][0]))*10, 0)))+values['Type'][0]+values['Sign'][0])
                                                else:
                                                    if single:
                                                        test.append(((values[names[Column0]]*10).astype(int)))
                                                    else:
                                                        test.append(np.concatenate(((values[names[Column1]]*10).astype(int), (values[names[Column2]]*10).astype(int))))
                                                    label1.append(values['Type'][0])
                                                    label2.append(values['Sign'][0])
                                                    label3.append(str(int(round((data[values['Type'][0]][0][0] - abs(values[values['Type'][0]][0]))*10, 0)))+values['Type'][0]+values['Sign'][0])
                                else:
                                    if values['Type'][0] == title1:
                                        if values['State'].any():
                                            if (round((data[values['Type'][0]][0][0] - abs(values[values['Type'][0]][0]))*10, 0)) % step == 0:
                                                if 0 < values['DataSet'][0] < set_:
                                                    if single:
                                                        train.append(((values[names[Column0]]*10).astype(int)))
                                                    else:
                                                        train.append(np.concatenate(((values[names[Column1]]*10).astype(int), (values[names[Column2]]*10).astype(int))))
                                                    labels1.append(values['Type'][0])
                                                    labels2.append(values['Sign'][0])
                                                    labels3.append(str(int(round((data[values['Type'][0]][0][0] - abs(values[values['Type'][0]][0]))*10, 0)))+values['Type'][0]+values['Sign'][0])
                                                else:
                                                    if single:
                                                        test.append(((values[names[Column0]]*10).astype(int)))
                                                    else:
                                                        test.append(np.concatenate(((values[names[Column1]]*10).astype(int), (values[names[Column2]]*10).astype(int))))
                                                    label1.append(values['Type'][0])
                                                    label2.append(values['Sign'][0])
                                                    label3.append(str(int(round((data[values['Type'][0]][0][0] - abs(values[values['Type'][0]][0]))*10, 0)))+values['Type'][0]+values['Sign'][0])

                        train   = np.array(train)
                        test    = np.array(test)
                        labels1 = np.array(labels1)
                        label1  = np.array(label1)
                        labels2 = np.array(labels2)
                        label2  = np.array(label2)
                        labels3 = np.array(labels3)
                        label3  = np.array(label3)

                        # print "\nPredicting the Movement Type"
                        y_pred1  = gnb.fit(train, labels1).predict(test)
                        y_pred1z = zip(y_pred1, label1)
                        # print("Number of mislabeled points out of a total %d points : %d" % (test.shape[0], np.array(label1 != y_pred1).sum()))
                        # print "Score = {0}% Success".format(gnb.score(test,label1)*100)

                        # print "\nPredicting the direction: Positive or Negative"
                        y_pred2  = gnb.fit(train, labels2).predict(test)
                        y_pred2z = zip(y_pred2, label2)
                        # print("Number of mislabeled points out of a total %d points : %d" % (test.shape[0], np.array(label2 != y_pred2).sum()))
                        # print "Score = {0}% Success".format(gnb.score(test,label2)*100)

                        # print "\nPredicting the Rotation"
                        y_pred3  = gnb.fit(train, labels3).predict(test)
                        y_pred3z = zip([i[:-3] for i in y_pred3], [i[:-3] for i in label3], label1, label2)
                        # print("Number of mislabeled points out of a total %d points : %d" % (test.shape[0], np.array(label3 != y_pred3).sum()))
                        print "Score = {0}% Success".format(gnb.score(test,label3)*100)

                        SaveDataLine.append(gnb.score(test,label3))
                        # if set_ == 10 and step == 1: break
                        # print gnb.score(test,label3)
                        print set_-1, step
                    SaveDataArray.append(SaveDataLine)
                if All:
                    if single:
                        Name = "TSP_PinDataRotationTestRxRyPN_{0}_{1}.txt".format(ztool, names[Column0])
                    else:
                        Name = "TSP_PinDataRotationTestRxRyPN_{0}_{1}_{2}.txt".format(ztool, names[Column1], names[Column2])
                else:
                    if differentsigns:
                        if single:
                            Name = "TSP_PinDataRotationTest{0}{1}_{2}_{3}.txt".format(title1, title2, ztool, names[Column0])
                        else:
                            Name = "TSP_PinDataRotationTest{0}{1}_{2}_{3}_{4}.txt".format(title1, title2, ztool, names[Column1], names[Column2])
                    else:
                        if single:
                            Name = "TSP_PinDataRotationTest{0}PN_{1}_{2}.txt".format(title1, ztool, names[Column0])
                        else:
                            Name = "TSP_PinDataRotationTest{0}PN_{1}_{2}_{3}.txt".format(title1, ztool, names[Column1], names[Column2])
                print os.path.join(directory, Name)
                if savingtext:
                    rw.writeList2File(os.path.join(directory, Name), SaveDataArray)

                if Graph:
                    fig = plt.figure()
                    ax = plt.subplot(1,1,1)
                    plt.title(Name + '\nTraining Sets: {0}    Testing Sets: {1}    Step distance: {2}mm'.format(set_-1, Sets-1, float(step)/10))
                    plt.xlabel("Actual distance, label (Degrees)")
                    plt.ylabel("Predicted distance, label (Degrees)")

                    x1 = [float(i[1])/10 for i in y_pred3z]
                    y1 = [float(i[:-3])/10 for i in y_pred3]
                    labels3 = [float(i[:-3])/10 for i in label3]

                    best_fit = plt.plot(labels3, labels3, 'r-', label="Correct Classification")
                    Classifier_Output = plt.scatter(x1, y1, c='blue', marker="x", label="Classifier Output")
                    plt.annotate('Rotation Y', xy=(-5, -4), xytext=(-10, 0), arrowprops=dict(facecolor='black', shrink=0.2))
                    plt.annotate('Rotation X', xy=(5, 6), xytext=(0, 10), arrowprops=dict(facecolor='black', shrink=0.2))
                    handles, labels = ax.get_legend_handles_labels()
                    plt.legend(handles, labels, loc=4)
                    plt.grid()
                    if savingGraph:
                        plt.savefig(os.path.join(directory, Name[:-4]+'.png'), dpi=None, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None)
                    if show:
                        plt.show()




else:
    print "Excluding Sign"
    for title1 in titles1:
        for title2 in titles2:
    # title1 = titles1[0]
    # title2 = titles2[0]
    # if True:
            for single in range(2):
                SaveDataArray  = []
                for set_ in range(2, Sets): # for set_ in range(2, Sets):
                    SaveDataLine  = []
                    for step in range(1, Sets): # for step in range(1, Sets):
                        train, test, labels1, labels2, labels3, label1, label2, label3 = [], [], [], [], [], [], [], []
                        for values in data[['Displacement', 'DifferenceX', 'DifferenceY', 'Bearing', 'X', 'Y', 'Z', 'Rx', 'Ry', 'Rz', 'DataSet', 'State', 'Type', 'Sign']]:
                            if All:
                                if values['State'].any():
                                    if (round((data[values['Type'][0]][0][0] - abs(values[values['Type'][0]][0]))*10, 0)) % step == 0:
                                        if 0 < values['DataSet'][0] < set_:
                                            if single:
                                                train.append(((values[names[Column0]]*10).astype(int)))
                                            else:
                                                train.append(np.concatenate(((values[names[Column1]]*10).astype(int), (values[names[Column2]]*10).astype(int))))
                                            labels1.append(values['Type'][0])
                                            labels2.append(values['Sign'][0])
                                            labels3.append(str(int(round((data[values['Type'][0]][0][0] - abs(values[values['Type'][0]][0]))*10, 0))))
                                        else:
                                            if single:
                                                test.append(((values[names[Column0]]*10).astype(int)))
                                            else:
                                                test.append(np.concatenate(((values[names[Column1]]*10).astype(int), (values[names[Column2]]*10).astype(int))))
                                            label1.append(values['Type'][0])
                                            label2.append(values['Sign'][0])
                                            label3.append(str(int(round((data[values['Type'][0]][0][0] - abs(values[values['Type'][0]][0]))*10, 0))))
                            else:
                                if differentsigns:
                                    if (values['Type'][0] == title1) and (values['Sign'][0] == title2):
                                        if values['State'].any():
                                            if (round((data[values['Type'][0]][0][0] - abs(values[values['Type'][0]][0]))*10, 0)) % step == 0:
                                                if 0 < values['DataSet'][0] < set_:
                                                    if single:
                                                        train.append(((values[names[Column0]]*10).astype(int)))
                                                    else:
                                                        train.append(np.concatenate(((values[names[Column1]]*10).astype(int), (values[names[Column2]]*10).astype(int))))
                                                    labels1.append(values['Type'][0])
                                                    labels2.append(values['Sign'][0])
                                                    labels3.append(str(int(round((data[values['Type'][0]][0][0] - abs(values[values['Type'][0]][0]))*10, 0))))
                                                else:
                                                    if single:
                                                        test.append(((values[names[Column0]]*10).astype(int)))
                                                    else:
                                                        test.append(np.concatenate(((values[names[Column1]]*10).astype(int), (values[names[Column2]]*10).astype(int))))
                                                    label1.append(values['Type'][0])
                                                    label2.append(values['Sign'][0])
                                                    label3.append(str(int(round((data[values['Type'][0]][0][0] - abs(values[values['Type'][0]][0]))*10, 0))))
                                else:
                                    if values['Type'][0] == title1:
                                        if values['State'].any():
                                            if (round((data[values['Type'][0]][0][0] - abs(values[values['Type'][0]][0]))*10, 0)) % step == 0:
                                                if 0 < values['DataSet'][0] < set_:
                                                    if single:
                                                        train.append(((values[names[Column0]]*10).astype(int)))
                                                    else:
                                                        train.append(np.concatenate(((values[names[Column1]]*10).astype(int), (values[names[Column2]]*10).astype(int))))
                                                    labels1.append(values['Type'][0])
                                                    labels2.append(values['Sign'][0])
                                                    labels3.append(str(int(round((data[values['Type'][0]][0][0] - abs(values[values['Type'][0]][0]))*10, 0))))
                                                else:
                                                    if single:
                                                        test.append(((values[names[Column0]]*10).astype(int)))
                                                    else:
                                                        test.append(np.concatenate(((values[names[Column1]]*10).astype(int), (values[names[Column2]]*10).astype(int))))
                                                    label1.append(values['Type'][0])
                                                    label2.append(values['Sign'][0])
                                                    label3.append(str(int(round((data[values['Type'][0]][0][0] - abs(values[values['Type'][0]][0]))*10, 0))))

                        train   = np.array(train)
                        test    = np.array(test)
                        labels1 = np.array(labels1)
                        label1  = np.array(label1)
                        labels2 = np.array(labels2)
                        label2  = np.array(label2)
                        labels3 = np.array(labels3)
                        label3  = np.array(label3)

                        # print "\nPredicting the Movement Type"
                        y_pred1  = gnb.fit(train, labels1).predict(test)
                        y_pred1z = zip(y_pred1, label1)
                        # print("Number of mislabeled points out of a total %d points : %d" % (test.shape[0], np.array(label1 != y_pred1).sum()))
                        # print "Score = {0}% Success".format(gnb.score(test,label1)*100)

                        # print "\nPredicting the direction: Positive or Negative"
                        y_pred2  = gnb.fit(train, labels2).predict(test)
                        y_pred2z = zip(y_pred2, label2)
                        # print("Number of mislabeled points out of a total %d points : %d" % (test.shape[0], np.array(label2 != y_pred2).sum()))
                        # print "Score = {0}% Success".format(gnb.score(test,label2)*100)

                        # print "\nPredicting the Rotation"
                        y_pred3  = gnb.fit(train, labels3).predict(test)
                        y_pred3z = zip(y_pred3, label3, label1, label2)
                        # print("Number of mislabeled points out of a total %d points : %d" % (test.shape[0], np.array(label3 != y_pred3).sum()))
                        # print "Score = {0}% Success".format(gnb.score(test,label3)*100)

                        SaveDataLine.append(gnb.score(test,label3))
                        # if set_ == 10 and step == 1: break
                        print set_-1, step
                    SaveDataArray.append(SaveDataLine)
                if All:
                    if single:
                        Name = "TSP_PinDataRotationTestRxRyPN_{0}_{1}.txt".format(ztool, names[Column0])
                    else:
                        Name = "TSP_PinDataRotationTestRxRyPN_{0}_{1}_{2}.txt".format(ztool, names[Column1], names[Column2])
                else:
                    if differentsigns:
                        if single:
                            Name = "TSP_PinDataRotationTest{0}{1}_{2}_{3}.txt".format(title1, title2, ztool, names[Column0])
                        else:
                            Name = "TSP_PinDataRotationTest{0}{1}_{2}_{3}_{4}.txt".format(title1, title2, ztool, names[Column1], names[Column2])
                    else:
                        if single:
                            Name = "TSP_PinDataRotationTest{0}PN_{1}_{2}.txt".format(title1, ztool, names[Column0])
                        else:
                            Name = "TSP_PinDataRotationTest{0}PN_{1}_{2}_{3}.txt".format(title1, ztool, names[Column1], names[Column2])

                print os.path.join(directory, Name)
                if saving:
                    rw.writeList2File(os.path.join(directory, Name), SaveDataArray)

                if Graph:
                    fig = plt.figure()
                    ax = plt.subplot(1,1,1)
                    plt.title(Name + '\nTraining Sets: {0}    Testing Sets: {1}    Step distance: {2}mm'.format(set_-1, Sets-1, float(step)/10))
                    plt.xlabel("Actual distance, label (Degrees)")
                    plt.ylabel("Predicted distance, label (Degrees)")

                    x1 = [float(i[1])/10 for i in y_pred3z]
                    y1 = [float(i)/10 for i in y_pred3]
                    labels3 = [float(i)/10 for i in label3]

                    best_fit = plt.plot(labels3, labels3, 'r-', label="Correct Classification")
                    Classifier_Output = plt.scatter(x1, y1, c='blue', marker="x", label="Classifier Output")
                    handles, labels = ax.get_legend_handles_labels()
                    plt.legend(handles, labels, loc=4)
                    plt.grid()
                    if saving:
                        plt.savefig(os.path.join(directory, Name[:-4]+'.png'), dpi=None, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None)
                    if show:
                        plt.show()
