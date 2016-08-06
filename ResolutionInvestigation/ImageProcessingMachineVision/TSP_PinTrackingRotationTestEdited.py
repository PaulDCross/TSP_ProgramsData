# Standard imports
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../libraries/MachineVisionAndmore")
from PillowEdited import *
import cv2
import numpy as np
import math
import time
import copy
from itertools import islice
from operator import itemgetter
from scipy.interpolate import interp1d
from scipy import ndimage

def makedir(DIR):
    if not os.path.exists(DIR):
        os.makedirs(DIR)
        time.sleep(0.5)

np.set_printoptions(precision=3, suppress=True)

rw             = rw()
Display        = 0
Record         = 0
SaveNumpy      = 0
SaveIndividual = 1
extrnl         = 0
# pathname1      = os.path.join("TSP_Pictures", "NewPillowRotationTest", "RotationTest167.5", "350.0mm", "01", "Rx", "P", "Internal", "%003d" % 1) + ".png"
# FirstImage     = cv2.imread(pathname1)
# refPt          = click_and_crop().crop_main(FirstImage)
# refPt          = [(22, 87), (1146, 509)]
# refPt          = [(429, 83), (783, 616)]
refPt          = [(124, 83), (1057, 585)]
x1, y1         = refPt[0][0], refPt[0][1]
x2, y2         = refPt[1][0], refPt[1][1]
colour         = [1]
A              = 0
ztool          = 167.5
zcoordinate    = 350.0
Sign           = ['P', 'N']
DIR            = os.path.join("TSP_Pictures", "NewPillowRotationTest", "RotationTest{0}".format(ztool), "{0}mm".format(zcoordinate))
numFolders     = 2#int(1 + len([name for name in os.listdir(DIR) if os.path.isdir(os.path.join(DIR, name))]))
Start          = 1

for fold in range(Start, numFolders):
    directory = os.path.join(DIR, "%02d" % fold)
    Types = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    # MovementType = os.path.join(directory, "Images")
    for Type in range(len([name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))])):
        # print Type
        MovementType  = os.path.join(directory, Types[Type])
        for sign in Sign:
            PictureFolder = os.path.join(MovementType, sign, "Internal")
            PictureFolderE = os.path.join(MovementType, sign, "External")

            CoM = []
            if Record:
                # Define the codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                out    = cv2.VideoWriter(os.path.join(MovementType, 'TSPMk2' + Types[Type] + sign + "I" + '.avi'),fourcc, 10.0, (2117,656))
                if extrnl:
                    # Define the codec and create VideoWriter object
                    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                    out2    = cv2.VideoWriter(os.path.join(MovementType, 'TSP' + Types[Type] + sign + "E" + '.avi'),fourcc, 30.0, (640,480))

            #############################################################################################

            with open(os.path.join(directory, "LogFile.txt")) as File:
                Textfile = [line for line in File.read().splitlines() if line]
                lineNums = [_ for _,s in enumerate(Textfile) if 'Starting Sequence:' in s]
            ls = []
            for _,i in enumerate(lineNums):
                if _+1 < 4:
                    with open(os.path.join(directory, "LogFile.txt")) as File:
                        resultlist = [line.split(":  [[")[-1].split("]] ")[0].split("], [")[0].split(", ") + line.split(":  [[")[-1].split("]] ")[0].split("], [")[1].split(", ") for line in islice(Textfile, lineNums[_]+5, lineNums[_+1], 3)]
                        resultlist.insert(0, [Textfile[lineNums[_]].split("\\")[-2], Textfile[lineNums[_]].split("\\")[-1][0][0]])
                        ls.append(resultlist)
                        if [Types[Type], sign] in resultlist:
                            index = len(ls)-1
                else:
                    with open(os.path.join(directory, "LogFile.txt")) as File:
                        resultlist = [line.split(":  [[")[-1].split("]] ")[0].split("], [")[0].split(", ") + line.split(":  [[")[-1].split("]] ")[0].split("], [")[1].split(", ") for line in islice(Textfile, lineNums[_]+5, None, 3)]
                        del resultlist[-1]
                        resultlist.insert(0, [Textfile[lineNums[_]].split("\\")[-2], Textfile[lineNums[_]].split("\\")[-1][0][0]])
                        ls.append(resultlist)
                        if [Types[Type], sign] in resultlist:
                            index = len(ls)-1

            #############################################################################################

            first = 1
            last  = (len([name for name in os.listdir(PictureFolder) if os.path.isfile(os.path.join(PictureFolder, name))]))

            pathname1  = os.path.join(PictureFolder, "%003d" % first) + ".png"
            FirstImage = cv2.imread(pathname1)
            # If the first picture is valid
            if (FirstImage.all()):
                # Setup the first image
                init          = Pillow(FirstImage, refPt)
                ROI1, _       = init.getFrame()
                # Sending the keypoints data to the class Pins in Pillow. Gives you the regions of the pins in coords.txt
                Columns, Rows, _ = Pins(refPt).main(init.detectorParameters().detect(ROI1))
                print Columns
                # Read the numbered regional data from the text file
                xyn           = rw.readFile2List("Pin_Regions.txt")
                # Find the coordinates of the pins in the first image
                data1         = init.initialiseData(xyn)

            for picture in range(first, last):
                pathname2    = os.path.join(PictureFolder, "%003d" % picture) + ".png"
                SecondImage  = cv2.imread(pathname2)
                frame        = copy.deepcopy(SecondImage)
                BlackImage   = np.zeros((frame.shape[0], x2-x1, 3), np.uint8)#; BlackImage.fill(255)
                BearingImage = copy.deepcopy(BlackImage[y1:y2, 0:x2-x1])
                pathname3    = os.path.join(PictureFolderE, "%003d" % picture) + ".png"
                extrnalImage = cv2.imread(pathname3)
                # print SecondImage.all()
                if (SecondImage.all()):
                    print fold, Types[Type], sign, first, picture
                    try:
                        # Set up the second image
                        rec                 = Pillow(frame, refPt)
                        ROI, frame_with_box = rec.getFrame()
                        # Set the detectors parametors and detect blobs.
                        keypoints           = rec.detectorParameters().detect(ROI)
                        Frame               = frame[y1:y2, x1:x2]
                        # Draw detected blobs as red circles. cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
                        Frame               = cv2.drawKeypoints(Frame, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                        data2               = rec.getDataSet2(keypoints, xyn)
                        DistanceBearing     = rec.measurements(data1, data2, len(keypoints))
                        DATA                = np.array([tuple(data) for data in [data1[i] + data2[i][1:] + DistanceBearing[i][1:] + [directory[-2:]] + [first] + [picture] + [float(DIR.split('\\')[-1][:-2])] + [ls[index][picture][0]] + [ls[index][picture][1]] + [ls[index][picture][2]] + [ls[index][picture][3]] + [ls[index][picture][4]] + [ls[index][picture][5]] + [Types[Type]] + [sign] for i in xrange(len(keypoints))]], dtype=[('Pin','i4'), ('OriginalXcoord','f4'), ('OriginalYcoord','f4'), ('OriginalPinSize','f4'), ('NewXcoord','f4'), ('NewYcoord','f4'), ('NewPinSize','f4'), ('State','bool'), ('DifferenceX','f4'), ('DifferenceY','f4'), ('Displacement','f4'), ('Bearing','f4'), ('DifferencePinSize','f4'), ('DataSet','i4'), ('PastImage','i4'), ('PresentImage','i4'), ('Height', 'f4'), ('X','f4'), ('Y','f4'), ('Z','f4'), ('Rx','f4'), ('Ry','f4'), ('Rz','f4'), ('Type', 'S4'), ('Sign', 'S4')])
                        if A == 0:
                            array1          = DATA
                            array2          = np.array(np.split(DATA, range(Columns,len(DATA),Columns)))
                            A += 1
                        else:
                            array1          = np.vstack((array1, DATA))
                            array2          = np.vstack((array2, np.array(np.split(DATA, range(Columns,len(DATA),Columns)))))
                        DATAarray           = np.array(np.split(DATA, range(Columns,len(DATA),Columns)))
                        if SaveIndividual:
                            Directory       = os.path.join(MovementType, "DataFiles" + PictureFolder.split("\\")[-3:-1][1])
                            makedir(Directory)
                            rw.writeList2File(os.path.join(Directory, "Data_%d.txt" % picture), DATA)

                        for data in DATA:
                            if data['State']: # Draw the Line
                                # Drawing the bearings
                                colour.append(data['DifferencePinSize'])
                                yy2         = 255
                                yy1         = 20
                                pinSizeX2   = 3.0
                                pinSizeX1   = 0.0
                                pinDistX2   = 10.0
                                pinDistX1   = 0.0
                                mPS         = ((yy2 - yy1) / (pinSizeX2 - pinSizeX1))
                                mPD         = ((yy2 - yy1) / (pinDistX2 - pinDistX1))
                                cv2.line(BearingImage, (int(data['OriginalXcoord']), int(data['OriginalYcoord'])), (int((data['OriginalXcoord']) + 10 * math.sin(math.radians(data['Bearing']))), int((data['OriginalYcoord']) + 10 * math.cos(math.radians(data['Bearing'])))), (mPD * abs(data['Displacement']) + yy1, mPD * abs(data['Displacement']) + yy1, mPD * abs(data['Displacement']) + yy1), 1)
                                cv2.line(BearingImage, (int(data['OriginalXcoord']), int(data['OriginalYcoord'])), (int((data['OriginalXcoord']) - 300 * math.sin(math.radians(data['Bearing']))), int((data['OriginalYcoord']) - 300 * math.cos(math.radians(data['Bearing'])))), (mPD * abs(data['Displacement']) + yy1, mPD * abs(data['Displacement']) + yy1, mPD * abs(data['Displacement']) + yy1), 1)
                                # cv2.circle(BearingImage, (int((data['OriginalXcoord']) - 100 * math.sin(math.radians(data['Bearing']))), int((data['OriginalYcoord']) - 100 * math.cos(math.radians(data['Bearing'])))), 1, (255,255,255))
                                cv2.circle(BearingImage, (int((data['OriginalXcoord']) + 10 * math.sin(math.radians(data['Bearing']))), int((data['OriginalYcoord']) + 10 * math.cos(math.radians(data['Bearing'])))), 1, (mPD * abs(data['Displacement']) + yy1, 0, 0), 1)
                                cv2.putText(BearingImage, "%.3f" % data['Displacement'], (int(data['OriginalXcoord']) - 14, int(data['OriginalYcoord']) - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, mPD * abs(data['Displacement']) + yy1, 0), 1)
                                cv2.putText(BearingImage, "%.3f" % data['DifferencePinSize'], (int(data['OriginalXcoord']) - 14, int(data['OriginalYcoord']) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, mPS * abs(data['DifferencePinSize']) + yy1), 1)
                                # Draw on the Image
                                cv2.putText(Frame, "%d" % data['Pin'], (int(data['NewXcoord']) - 7, int(data['NewYcoord']) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, 8)

                                cv2.line(Frame, (int(data['OriginalXcoord']), int(data['OriginalYcoord'])), (int(data['NewXcoord']), int(data['NewYcoord'])), (0, 0, 255), 2)
                                cv2.circle(Frame, (int(data['NewXcoord']), int(data['NewYcoord'])), 1, (0, 0, 255), 2)

                        frame_with_box[y1:y2, x1:x2] = Frame
                        # Creates a black image and sets each pixel value as white.
                        width = 60
                        whiteBar = np.zeros((width, frame_with_box.shape[1], 3), np.uint8); whiteBar.fill(255)
                        # Sets the region specified to be equal to the white image create above.
                        frame_with_box[0:width, 0:frame_with_box.shape[1]] = whiteBar
                        # Give the frame a title and display the number of blobs.
                        cv2.putText(frame_with_box, pathname2, (5, width-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        cv2.putText(frame_with_box, "Tracking %d pins" % DATA[-1][0], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                        BlackImage[y1:y2, 0:x2-x1] = BearingImage
                        textsize = cv2.getTextSize(str(ls[index][picture]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                        cv2.putText(BlackImage, str(ls[index][picture]), ((x2-x1)-textsize[0][0], width-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        video = np.concatenate((BlackImage, frame_with_box), axis=1)

                        # Show the frames
                        if Display:
                            cv2.imshow("Camera", video)
                            # cv2.imshow("Camera2", BearingImage)
                            # cv2.imshow("Camera3", frame_with_box)
                            # cv2.imshow("ROI", ROI)

                            if cv2.waitKey(1) & 0xFF == 27:
                                cv2.destroyAllWindows()
                                break
                        if Record:
                            out.write(video)
                            if extrnl:
                                out2.write(extrnalImage)

                    except (ValueError, IndexError):
                        print len(keypoints)
                        cv2.imshow("Frame", cv2.drawKeypoints(ROI, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
                        if cv2.waitKey(0) & 0xFF == 27:
                            cv2.destroyAllWindows()

            if Record:
                out.release()
                if extrnl:
                    out2.release()
if SaveNumpy:
    np.save(os.path.join(DIR, "Mk2dataline" + str(Start) + str(numFolders-1)), array1)
    np.save(os.path.join(DIR, "Mk2dataarray" + str(Start) + str(numFolders-1)), array2)
# np.load("DATA.npy")
