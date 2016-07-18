# Standard imports
from Pillow import *
import cv2
import numpy as np
import math
import time
import copy
from operator import itemgetter
from scipy.interpolate import interp1d
from scipy import ndimage

rw = rw()
# refPt = click_and_crop().crop_main()
# refPt = [(22, 97), (588, 429)]
refPt = [(124, 83), (1057, 585)]
# refPt = [(429, 83), (783, 616)]
# x1, y1 = refPt[0][0], refPt[0][1]
# x2, y2 = refPt[1][0], refPt[1][1]
x1, y1 = refPt[0][0], refPt[0][1]
x2, y2 = refPt[1][0], refPt[1][1]
colour = [1]

cam = cv2.VideoCapture(1)
cam.set(3, 1200)            # horizontal pixels
cam.set(4, 720)            # vertical pixels
time.sleep(1)
initialize = 1
while True:
    ret, image   = cam.read()
    # If the first picture is valid
    if ret:
        if initialize:
            # Setup the first image
            init          = Pillow(image, refPt)
            ROI1, _       = init.getFrame()
            # Sending the keypoints data to the class Pins in Pillow. Gives you the regions of the pins in coords.txt
            Columns, Rows = Pins(refPt).main(init.detectorParameters().detect(ROI1))
            # Read the numbered regional data from the text file
            xyn           = rw.readFile2List("Pin_Regions.txt")
            # Find the coordinates of the pins in the first image
            data1         = init.initialiseData(xyn)
            initialize    = 0
            X             = interp1d([0,8],[data1[0][1],data1[-1][1]])
            Y             = interp1d([0,11],[data1[0][2],data1[-1][2]])
            CoM           = []
            # hsv = np.zeros(((y2 - y1), (x2 - x1), 3), np.uint8)
            # hsv[...,1] = 255
        try:
            # Set up the second image
            rec                 = Pillow(image, refPt)
            ROI2,frame_with_box = rec.getFrame()
            # Set the detectors parametors and detect blobs.
            keypoints           = rec.detectorParameters().detect(ROI2)
            Frame               = image[y1:y2, x1:x2]
            # Draw detected blobs as red circles. cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
            Frame               = cv2.drawKeypoints(Frame, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            data2               = rec.getDataSet2(keypoints, xyn)
            DistanceBearing     = rec.measurements(data1, data2, len(keypoints))
            DATA                = [data1[i] + data2[i][1:] + DistanceBearing[i][1:] + xyn[i][:-1] for i in xrange(len(keypoints))]
            BlackImage   = np.zeros((image.shape[0], x2-x1, 3), np.uint8)#; BlackImage.fill(255)
            BearingImage = copy.deepcopy(BlackImage[y1:y2, 0:x2-x1])
            # Directory = os.path.join("ABBTest", "TSP_Pictures", "Datafiles", MovementType)
            # makedir(Directory)
            # rw.writeList2File(os.path.join(Directory, "Data_%d.txt" % picture), DATA)

            # colour = rec.colorMap(DistanceBearing)

            # Calculate the distance between the pins and display the distance as a change of colour.
            DATAsplit = chunker(DATA, Columns)
            [Splits.append(Splits[-1]) for Splits in DATAsplit]
            DATAsplit.append(DATAsplit[-1])
            DATAarray = np.array(DATAsplit)
            # if 1 in map(itemgetter(7), DATA):
            #     CoM.append(ndimage.measurements.center_of_mass(DATAarray[:,:,10]))
            #     if len(CoM) > 50:
            #         CoM.pop(0)

            for i in range(len(DATAarray[:]) - 1):            # Number of rows.
                for j in range(len(DATAarray[i][:]) - 1):     # Number of columns in each row.
                    # Calculate the Distance between the pins
                    d1y = math.sqrt((int(DATAarray[i + 1][j][1]) - int(DATAarray[i][j][1])) ** 2 + (int(DATAarray[i + 1][j][2]) - int(DATAarray[i][j][2])) ** 2)
                    d1x = math.sqrt((int(DATAarray[i][j + 1][1]) - int(DATAarray[i][j][1])) ** 2 + (int(DATAarray[i][j + 1][2]) - int(DATAarray[i][j][2])) ** 2)
                    d2y = math.sqrt((int(DATAarray[i + 1][j][4]) - int(DATAarray[i][j][4])) ** 2 + (int(DATAarray[i + 1][j][5]) - int(DATAarray[i][j][5])) ** 2)
                    d2x = math.sqrt((int(DATAarray[i][j + 1][4]) - int(DATAarray[i][j][4])) ** 2 + (int(DATAarray[i][j + 1][5]) - int(DATAarray[i][j][5])) ** 2)
                    # colour.append((d2x-d1x)); colour.append((d2y-d1y))
                    # m = ((255 - 100) / (max(colour) - min(colour)))
                    # cv2.line(Frame, (int(DATAarray[i][j][4]), int(DATAarray[i][j][5])), (int(DATAarray[i + 1][j][4]), int(DATAarray[i + 1][j][5])), (0, m * abs(d2y - d1y) + 100, 0), 5)
                    # cv2.line(Frame, (int(DATAarray[i][j][4]), int(DATAarray[i][j][5])), (int(DATAarray[i][j + 1][4]), int(DATAarray[i][j + 1][5])), (0, m * abs(d2x - d1x) + 100, 0), 5)
                    # cv2.rectangle(BearingImage, (int(DATAarray[i-1][j][13]), int(DATAarray[i-1][j][14])), (int(DATAarray[i][j-1][13]), int(DATAarray[i][j-1][14])), (255,255,255), -1)

            for data in DATA:
                # Drawing the bearings
                colour.append(data[12])
                # pinSizeX2 = max(colour)
                # pinSizeX1 = min(colour)
                yy2       = 255
                yy1       = 20
                pinSizeX2 = 2.0
                pinSizeX1 = 0.0
                pinDistX2 = 50
                pinDistX1 = 0.0
                mPS       = ((yy2 - yy1) / (pinSizeX2 - pinSizeX1))
                mPD       = ((yy2 - yy1) / (pinDistX2 - pinDistX1))
                cv2.line(BearingImage, (int(data[1]), int(data[2])), (int((data[1]) + 20 * math.sin(math.radians(data[11]))), int((data[2]) + 20 * math.cos(math.radians(data[11])))), (mPD * abs(data[10]) + yy1, mPD * abs(data[10]) + yy1, mPD * abs(data[10]) + yy1), 1)
                cv2.line(BearingImage, (int(data[1]), int(data[2])), (int((data[1]) - 200 * math.sin(math.radians(data[11]))), int((data[2]) - 200 * math.cos(math.radians(data[11])))), (mPD * abs(data[10]) + yy1, mPD * abs(data[10]) + yy1, mPD * abs(data[10]) + yy1), 1)
                cv2.circle(BearingImage, (int((data[1]) + 10 * math.sin(math.radians(data[11]))), int((data[2]) + 10 * math.cos(math.radians(data[11])))), 1, (mPD * abs(data[10]) + yy1, 0, 0), 1)
                cv2.putText(BearingImage, "%.1f" % data[10], (int(data[1]) - 14, int(data[2]) - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, mPD * abs(data[10]) + yy1, 0), 1)
                cv2.putText(BearingImage, "%.2f" % data[12], (int(data[1]) - 14, int(data[2]) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, mPS * abs(data[12]) + yy1), 1)
                # Draw on the Image
                cv2.putText(Frame, "%d" % data[0], (int(data[4]) - 7, int(data[5]) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, 8)
                if data[7]:
                    # Draw the Line
                    cv2.line(Frame, (int(data[1]), int(data[2])), (int(data[4]), int(data[5])), (0, 0, 255), 2)
                    cv2.circle(Frame, (int(data[4]), int(data[5])), 1, (0, 0, 255), 2)
            frame_with_box[y1:y2, x1:x2] = Frame
            # Creates a black image and sets each pixel value as white.
            width = 60
            whiteBar = np.zeros((width, int(np.shape(frame_with_box)[1]), 3), np.uint8); whiteBar.fill(255)
            # Sets the region specified to be equal to the white image create above.
            frame_with_box[0:width, 0:int(np.shape(frame_with_box)[1])] = whiteBar
            # Give the frame a title and display the number of blobs.
            # cv2.putText(frame_with_box, pathname2, (5, width-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(frame_with_box, "Tracking %d pins" % DATA[-1][0], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            # for xCoM, yCoM in CoM:
            #     cv2.circle(BearingImage, (Y(yCoM), X(xCoM)), 5, (0,255,0), 2)
            BlackImage[y1:y2, 0:x2-x1] = BearingImage
            video = np.concatenate((BlackImage, frame_with_box), axis=1)
            # Show the frames
            # cv2.imshow('frame2',bgr)
            cv2.imshow("Camera", video)
            # cv2.imshow("Overview", frame_with_box)
            # cv2.imshow("Bearings", BearingImage)
        except IndexError:
            initialize = 1



    k = cv2.waitKey(75) & 0xFF
    if k == 32:
        initialize = 1
    if k == 27:
        cv2.destroyAllWindows()
        break
