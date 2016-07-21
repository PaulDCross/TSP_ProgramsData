import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../libraries/MachineVisionAndmore")
from Pillow import rw
import numpy as np
import time
from matplotlib import pyplot as plt
from sklearn import linear_model
from operator import itemgetter
from itertools import groupby
np.set_printoptions(precision=3, suppress=True, linewidth = 150)

class rw():
    def readFile2List(self, textFile):
        with open(textFile, "r") as file:
            data = []
            for line in file.readlines():
                data.append([float(i) for i in line.split()])
        return data

    def writeList2File(self, textFile, DATA):
        with open(textFile, "w") as file:
            DATA = '\n'.join('\t'.join(map(str,j)) for j in DATA)
            file.write(DATA)

def makedir(DIR):
    if not os.path.exists(DIR):
        os.makedirs(DIR)
        time.sleep(0.5)

ProgramsData= os.path.join("..", "..", "..", "Python", "TSP_Testing", "TSP_Testing", "ProgramsData")
ztool       = 167.5
zcoordinate = 358.0
directory   = os.path.join("..", "..", "PinDataResults", "NewPillowRepeatabilityTestTypeResetBayesianRidge", "EE{0}".format(ztool))
makedir(directory)
DIR         = os.path.join(ProgramsData, "TSP_Pictures", "NewPillowRepeatabilityTest", "EE{0}".format(ztool), "{0}mm".format(zcoordinate))
# data      = np.load(os.path.join(DIR, "Otsudataline110.npy"))
data        = np.load(os.path.join(DIR, "dataline110.npy"))
print "Loaded Data"
names       = data.dtype.names
Column0     = names.index('Displacement')
Column1     = names.index('DifferenceX')
Column2     = names.index('DifferenceY')
saving      = 0
savingGraph = 1
Sets        = int(1 + len([name for name in os.listdir(DIR) if os.path.isdir(os.path.join(DIR, name))]))
for single in range(2):
    SaveDataArray = []
    for set_ in range(2, 3): # for set_ in range(2, Sets):
        SaveDataLine  = []
        for step in range(1, 2): # for step in range(1, Sets):
            gnb = linear_model.BayesianRidge(compute_score = True)
            train, test, labels1, labels2, label1, label2 = [], [], [], [], [], []
            for _, values in enumerate(data[['Displacement', 'DifferenceX', 'DifferenceY', 'Bearing', 'X', 'Y', 'Z', 'Rx', 'Ry', 'Rz', 'DataSet', 'State', 'Type', 'Depth']]):
                if values['Type'][0] == 'Z':
                    if np.count_nonzero(values['State']) > 4:
                        if (round((zcoordinate - abs(values['Z'][0]))*10, 0)) % step == 0:
                            if 0 < values['DataSet'][0] < set_:
                                if single:
                                    train.append(np.concatenate(((values[names[Column0]]*10).astype(int), )))
                                else:
                                    train.append(np.concatenate(((values[names[Column1]]*10).astype(int), (values[names[Column2]]*10).astype(int))))
                                labels1.append(values['Type'][0])
                                labels2.append(int(round((zcoordinate - abs(values['Z'][0]))*10, 0)))
                            else:
                                if single:
                                    test.append(np.concatenate(((values[names[Column0]]*10).astype(int), )))
                                else:
                                    test.append(np.concatenate(((values[names[Column1]]*10).astype(int), (values[names[Column2]]*10).astype(int))))
                                label1.append(values['Type'][0])
                                label2.append(int(round((zcoordinate - abs(values['Z'][0]))*10, 0)))
            print "\nSetting up train and test sets."
            train   = np.array(train)
            test    = np.array(test)
            labels1 = np.array(labels1)
            label1  = np.array(label1)
            labels2 = np.array(labels2)
            label2  = np.array(label2)
            print len(train), len(test)
            # print "Predicting the Movement Type"
            # y_pred1  = gnb.fit(train, labels1).predict(test)
            # y_pred1z = zip(y_pred1, label1)
            # print("Number of mislabeled points out of a total %dpoints : %d" % (test.shape[0], np.array(label1 != y_pred1).sum()))
            # print "Score = {0}% Success".format(gnb.score(test,label1)*100)

            print "Predicting the Depth"
            y_pred2            = gnb.fit(train, labels2).predict(test)
            y_pred2z           = zip(y_pred2, label2)
            predictions        = sorted(y_pred2z, key=itemgetter(1))
            # Group the predictions based on the actual depth
            groupedPredictions = np.array([list(j) for i,j in groupby(map(list,predictions), itemgetter(1))])
            # Calculate the mean and mad of the difference between the values
            madPredictions     = np.array([np.mean(abs(     np.subtract(    abs(np.subtract([c[0] for c in b], [c[1] for c in b]))  ,   np.mean(abs(np.subtract([c[0] for c in b], [c[1] for c in b]))))    )) for b in groupedPredictions])
            meanPredictions    = np.array([np.mean(abs(np.subtract([c[0] for c in b], [c[1] for c in b]))) for b in groupedPredictions])
            stdPredictions     = np.array([np.std(abs(np.subtract([c[0] for c in b], [c[1] for c in b]))) for b in groupedPredictions])
            xValues = [float(b[0][1])/10 for b in groupedPredictions]

            # print("Number of mislabeled points out of a total %d points : %d" % (test.shape[0], np.array(label2 != y_pred2).sum()))
            print "Score = {0}%, {1}mm, #{2}".format(round(gnb.score(test,label2)*100, 3), float(step)/10, set_-1)
            # print y_pred2z
            SaveDataLine.append(gnb.score(test,label2))
        SaveDataArray.append(SaveDataLine)
    if single:
        Name = "TSP_Repeat_DisplResetBayesianRidge"
    else:
        Name = "TSP_Repeat_DistX_DistYResetBayesianRidge"
    # if single:
    #     Name = "TSP_Repeatability_Displacement_Otsu"
    # else:
    #     Name = "TSP_Repeatability_DistanceX_DistanceY_Otsu"
    if saving:
        rw().writeList2File(os.path.join(directory, Name + ".txt"), SaveDataArray)

    if savingGraph:
        fig = plt.figure()
        ax  = plt.subplot(1,1,1)
        plt.title(Name + '\nTraining Sets: {0}    Testing Sets: {1}    Step distance: {2}mm'.format(set_-1, Sets-set_, float(step)/10))
        plt.xlabel("Actual depth from start position, (mm)")
        plt.ylabel("Predicted depth from start position, (mm)")
        x1                = [float(i[1])/10 for i in y_pred2z]
        y1                = [float(i[0])/10 for i in y_pred2z]
        labels2           = [float(i)/10 for i in label2]
        # toMatlab          = zip(x1, y1, labels2)
        # best_fit          = plt.plot(labels2, labels2, 'r-', label="Correct Classification")
        # Classifier_Output = plt.scatter(x1, y1, c='blue', marker="x", label="Classifier Output")
        MAD  = plt.plot(xValues, madPredictions, label="MAD of the difference between actual and predicted")
        mean = plt.plot(xValues, meanPredictions, label="Mean difference between actual and predicted")
        # std  = plt.plot(xValues, stdPredictions, label="Std")
        handles, labels   = ax.get_legend_handles_labels()
        # fig = plt.figure()
        # ax  = plt.subplot(1,1,1)
        # plt.plot([np.std(i) for i in y_pred2z])
        # rw().writeList2File(os.path.join(directory, Name + "_ML.txt"), toMatlab)
        # print "Saved for Matlab"
        plt.legend(handles, labels, loc=1)
        plt.grid()
        # plt.savefig(os.path.join(directory, Name + '.png'), dpi=None, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None)
        plt.show()
