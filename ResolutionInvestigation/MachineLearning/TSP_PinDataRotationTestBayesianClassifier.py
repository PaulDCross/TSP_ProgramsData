import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../libraries/MachineVisionAndmore")
from PillowEdited import rw
import numpy as np
import time
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
np.set_printoptions(precision=3, suppress=True, linewidth = 150)

def makedir(DIR):
    if not os.path.exists(DIR):
        os.makedirs(DIR)
        time.sleep(0.5)

ProgramsData = os.path.join("..", "..", "..", "Python", "TSP_Testing", "TSP_Testing", "ProgramsData")
ztool        = 167.5
zcoordinate  = 342.5
directory    = os.path.join(ProgramsData, "PinDataResults", "NewPillowRotationTestsTypeSignReset", "RotationTest{0}".format(ztool))
makedir(directory)
DIR          = os.path.join(ProgramsData, "TSP_Pictures", "RotationTests", "RotationTest{0}".format(ztool), "{0}mm".format(zcoordinate))
# data       = np.load(os.path.join(DIR, "Otsudataline110.npy"))
data         = np.load(os.path.join(DIR, "dataline110.npy"))
# data       = np.load(os.path.join(DIR, "dataline110.npy"))
print "Loaded Data"
names        = data.dtype.names
Column0      = names.index('Displacement')
Column1      = names.index('DifferenceX')
Column2      = names.index('DifferenceY')
savingtext   = 0
savingGraph  = 1
Sets         = int(1 + len([name for name in os.listdir(DIR) if os.path.isdir(os.path.join(DIR, name))]))
for single in range(2):
    SaveDataArray = []
    for set_ in range(10, 11): # for set_ in range(2, Sets):
        SaveDataLine  = []
        for step in range(1, 2): # for step in range(1, Sets):
            gnb = GaussianNB()
            train, test, labels1, labels2, labels3, label1, label2, label3 = [], [], [], [], [], [], [], []
            for _, values in enumerate(data[['Displacement', 'DifferenceX', 'DifferenceY', 'Bearing', 'X', 'Y', 'Z', 'Rx', 'Ry', 'Rz', 'DataSet', 'State', 'Type', 'Sign']]):
                if values['Type'][0] == 'Rx':
                    if np.count_nonzero(values['State']) > 4:
                        if (round((data[values['Type'][0]][0][0] - abs(values[values['Type'][0]][0]))*10, 0)) % step == 0:
                            if 0 < values['DataSet'][0] < set_:
                                if single:
                                    train.append(np.concatenate(((values[names[Column0]]*10).astype(int), )))
                                else:
                                    train.append(np.concatenate(((values[names[Column1]]*10).astype(int), (values[names[Column2]]*10).astype(int))))
                                labels1.append(values['Type'][0])
                                labels2.append(values['Sign'][0])
                                labels3.append(str(int(round((data[values['Type'][0]][0][0] - abs(values[values['Type'][0]][0]))*10, 0))*np.sign(values[values['Type'][0]][0])*-1)+values['Type'][0]+values['Sign'][0])
                            else:
                                if single:
                                    test.append(np.concatenate(((values[names[Column0]]*10).astype(int), )))
                                else:
                                    test.append(np.concatenate(((values[names[Column1]]*10).astype(int), (values[names[Column2]]*10).astype(int))))
                                label1.append(values['Type'][0])
                                label2.append(values['Sign'][0])
                                label3.append(str(int(round((data[values['Type'][0]][0][0] - abs(values[values['Type'][0]][0]))*10, 0))*np.sign(values[values['Type'][0]][0])*-1)+values['Type'][0]+values['Sign'][0])
            print "\nSetting up train and test sets."
            train   = np.array(train)
            test    = np.array(test)
            labels1 = np.array(labels1)
            label1  = np.array(label1)
            labels2 = np.array(labels2)
            label2  = np.array(label2)
            labels3 = np.array(labels3)
            label3  = np.array(label3)
            print len(train), len(test)
            print "Predicting the Rotation"
            y_pred3  = gnb.fit(train, labels3).predict(test)
            y_pred3z = zip([i[:-3] for i in y_pred3], [i[:-3] for i in label3], label1, label2)
            print "Trained Classifier"
            # print("Number of mislabeled points out of a total %d points : %d" % (test.shape[0], np.array(label3 != y_pred3).sum()))
            print "Score = {0}%, {1}mm, #{2}".format(round(gnb.score(test,label3)*100, 3), float(step)/10, set_-1)
            # print y_pred3z
            SaveDataLine.append(gnb.score(test,label3))
            gnb = False
        SaveDataArray.append(SaveDataLine)
    # if single:
    #     Name = "TSP_Rotation_Displacement"
    # else:
    #     Name = "TSP_Rotation_DistanceX_and_DistanceY"
    if single:
        Name = "Using Pin Displacement for the features of the Classifier"
    else:
        Name = "Using Delta X and Delta Y for the features of the Classifier"
    # if single:
    #     Name = "TSP_Rotation_Displacement_Otsu"
    # else:
    #     Name = "TSP_Rotation_DistanceX_DistanceY_Otsu"
    print os.path.join(directory, Name)
    if savingtext:
        rw().writeList2File(os.path.join(directory, Name + ".txt"), SaveDataArray)
    print SaveDataArray

    if savingGraph:
        fig = plt.figure()
        ax  = plt.subplot(1,1,1)
        plt.title(Name + '\nTraining Sets: {0}    Testing Sets: {1}    Step distance: {2}mm'.format(set_-1, Sets-set_, float(step)/10))
        plt.xlabel("Actual Angle, (Degrees)")
        plt.ylabel("Predicted Angle, (Degrees)")

        major_ticks = np.arange(-12, 12, 1)
        minor_ticks = np.arange(-12, 12, 0.2)

        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)

        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)

        ax.grid(which='major', alpha=1)

        # ax.grid(which='minor', alpha=0.5)
        # ax.set_xlim(-2,2)
        # ax.set_ylim(-2,2)

        x1                = [float(i[1])/10 for i in y_pred3z]
        y1                = [float(i[:-3])/10 for i in y_pred3]
        axlabels3         = [float(i[:-3])/10 for i in label3]
        toMatlab          = zip(x1, y1, axlabels3)
        best_fit          = plt.plot(axlabels3, axlabels3, 'r-', label="Correct Value")
        Error             = plt.bar(x1, abs(np.subtract(x1, y1)), 0.1, label="Error")
        Classifier_Output = plt.scatter(x1, y1, c='blue', marker="x", label="Machine Learning Output")
        handles, labels   = ax.get_legend_handles_labels()
        # rw().writeList2File(os.path.join(directory, Name + "_ML.txt"), toMatlab)
        # print "Saved for Matlab"
        # plt.annotate('Rotation Y', xy=(-5, -4), xytext=(-10, 0), arrowprops=dict(facecolor='black', shrink=0.2))
        # plt.annotate('Rotation X', xy=(5, 6), xytext=(0, 10), arrowprops=dict(facecolor='black', shrink=0.2))
        plt.legend(handles, labels, loc=2)
        # plt.savefig(os.path.join(Name + '.png'), dpi=None, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None)
        plt.show()




