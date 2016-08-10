# Standard imports
import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../libraries/MachineVisionAndmore")
from PillowOptimised import *
import cv2
import numpy as np
import numpy.ma as ma
import math
import time
import copy
np.set_printoptions(precision=3, suppress=True, linewidth = 150)
# from operator import itemgetter
# from scipy.interpolate import interp1d
# from scipy import ndimage
def makedir(DIR):
    if not os.path.exists(DIR):
        os.makedirs(DIR)
        time.sleep(0.5)

directory = 'Videos'

makedir(directory)
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
cam.set(4, 720)             # vertical pixels
time.sleep(1)
initialize = 1
record, recording = 0, 0
while True:
    ret, image   = cam.read()
    # If the first picture is valid
    if ret:
        if record:
            print "Setting up video"
            # Define the codec and create VideoWriter object
            fps               = 10
            numberofVideos    = len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])
            fourcc            = cv2.VideoWriter_fourcc(*'DIVX')
            VideoFrame        = cv2.VideoWriter(os.path.join(directory, 'RealTime_{0}FPS_{1}'.format(fps, numberofVideos) + '.avi'), fourcc, fps, (933,654))
            record, recording = 0, 1
            print fps, numberofVideos
            print "Recording"
        if initialize:
            print "Initialising"
            # Setup the first image
            # image1 = image
            init               = Pillow(image, refPt)
            ROI1, _            = init.getFrame()
            # cv2.imshow("Camera", ROI1)
            # Sending the keypoints data to the class Pins in Pillow. Gives you the regions of the pins in coords.txt
            # Read the numbered regional data from the text file
            Columns, Rows, xyn = Pins(refPt).main(init.detectorParameters().detect(ROI1))
            # Find the coordinates of the pins in the first image
            data1              = init.initialiseData(xyn)
            initialize         = 0
            superimposeLines   = 0
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
            DATA                = np.array([tuple(data) for data in [data1[i] + data2[i][1:] + DistanceBearing[i][1:] for i in xrange(len(keypoints))]], dtype=[('Pin Number', 'f4'), ('Reference X Coordinate', 'f4'), ('Reference Y Coordinate', 'f4'), ('Reference Pin Diameter', 'f4'), ('New X Coordinate', 'f4'), ('New Y Coordinate', 'f4'), ('New Pin Diameter', 'f4'), ('State', 'bool'), ('DifferenceX','f4'), ('DifferenceY','f4'), ('Displacement','f4'), ('Bearing','f4'), ('DifferencePinSize','f4')])#  + xyn[i][:-1]
            BlackImage          = np.zeros((image.shape[0], x2-x1, 3), np.uint8)#; BlackImage.fill(255)
            BearingImage        = copy.deepcopy(BlackImage[y1:y2, 0:x2-x1])

            mask                = [np.array([data[7]])*7 for data in DATA]
            findCentre          = np.array([np.array(data)[['Pin Number', 'Reference X Coordinate', 'Reference Y Coordinate', 'DifferenceX', 'DifferenceY', 'Displacement', 'Bearing']] for data in DATA])
            findCentre          = findCentre.compress(np.ravel(mask))
            # if len(findCentre) > 4:
            #     print np.mean(findCentre['Reference X Coordinate'] + findCentre['DifferenceX']), np.mean(findCentre['Reference Y Coordinate'] + findCentre['DifferenceY'])
            #     cv2.circle(BearingImage, (int(np.mean(findCentre['Reference X Coordinate'] + findCentre['DifferenceX'])), int(np.mean(findCentre['Reference Y Coordinate'] + findCentre['DifferenceY']))), 5, (0, 255, 255), -1)
            #     cv2.circle(BearingImage, (int(np.mean(findCentre['Reference X Coordinate'] + findCentre['DifferenceX'])), int(np.mean(findCentre['Reference Y Coordinate'] + findCentre['DifferenceY']))), 20, (0, 255, 255), 2)


            # coordinates = np.array([keypoints[i-1].pt for i in xrange(len(keypoints))])
            # p1, st, err = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), coordinates, None, winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            # # Select good points
            # good_new = p1[st==1]
            # good_old = p0[st==1]

            # # draw the tracks
            # for i, (new, old) in enumerate(zip(good_new, good_old)):
            #     a, b  = new.ravel()
            #     c, d  = old.ravel()
            #     mask  = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            #     frame = cv2.circle(frame_with_box, (a, b), 5, color[i].tolist(), -1)
            # img = cv2.add(frame, mask)


            # # Calculate the distance between the pins and display the distance as a change of colour.
            # DATAsplit = chunker(DATA, Columns)
            # [Splits.append(Splits[-1]) for Splits in DATAsplit]
            # DATAsplit.append(DATAsplit[-1])
            # DATAarray = np.array(DATAsplit)

            # for i in range(len(DATAarray[:]) - 1):            # Number of rows.
            #     for j in range(len(DATAarray[i][:]) - 1):     # Number of columns in each row.
            #         # Calculate the Distance between the pins
            #         d1y = math.sqrt((int(DATAarray[i + 1][j][1]) - int(DATAarray[i][j][1])) ** 2 + (int(DATAarray[i + 1][j][2]) - int(DATAarray[i][j][2])) ** 2)
            #         d1x = math.sqrt((int(DATAarray[i][j + 1][1]) - int(DATAarray[i][j][1])) ** 2 + (int(DATAarray[i][j + 1][2]) - int(DATAarray[i][j][2])) ** 2)
            #         d2y = math.sqrt((int(DATAarray[i + 1][j][4]) - int(DATAarray[i][j][4])) ** 2 + (int(DATAarray[i + 1][j][5]) - int(DATAarray[i][j][5])) ** 2)
            #         d2x = math.sqrt((int(DATAarray[i][j + 1][4]) - int(DATAarray[i][j][4])) ** 2 + (int(DATAarray[i][j + 1][5]) - int(DATAarray[i][j][5])) ** 2)
            #         colour.append((d2x-d1x)); colour.append((d2y-d1y))
            #         m = ((255 - 100) / (max(colour) - min(colour)))
            #         cv2.line(Frame, (int(DATAarray[i][j][4]), int(DATAarray[i][j][5])), (int(DATAarray[i + 1][j][4]), int(DATAarray[i + 1][j][5])), (0, m * abs(d2y - d1y) + 100, 0), 5)
            #         cv2.line(Frame, (int(DATAarray[i][j][4]), int(DATAarray[i][j][5])), (int(DATAarray[i][j + 1][4]), int(DATAarray[i][j + 1][5])), (0, m * abs(d2x - d1x) + 100, 0), 5)
            #         # cv2.rectangle(BearingImage, (int(DATAarray[i-1][j][13]), int(DATAarray[i-1][j][14])), (int(DATAarray[i][j-1][13]), int(DATAarray[i][j-1][14])), (255,255,255), -1)

            for data in DATA:
                # Drawing the bearings
                colour.append(data[12])
                # pinSizeX2 = max(colour)
                # pinSizeX1 = min(colour)
                yy2       = 255
                yy1       = 20
                pinSizeX2 = 3.0
                pinSizeX1 = 0.0
                pinDistX2 = 50
                pinDistX1 = 0.0
                mPS       = ((yy2 - yy1) / (pinSizeX2 - pinSizeX1))
                mPD       = ((yy2 - yy1) / (pinDistX2 - pinDistX1))
                if superimposeLines:
                    cv2.line(BearingImage, (int(data[1]), int(data[2])), (int((data[1]) - 200 * math.sin(math.radians(data[11]))), int((data[2]) - 200 * math.cos(math.radians(data[11])))), (mPD * abs(data[10]) + yy1, mPD * abs(data[10]) + yy1, mPD * abs(data[10]) + yy1), 1)
                cv2.line(BearingImage, (int(data[1]), int(data[2])), (int((data[1]) + 30 * math.sin(math.radians(data[11]))), int((data[2]) + 30 * math.cos(math.radians(data[11])))), (mPD * abs(data[10]) + yy1, mPD * abs(data[10]) + yy1, mPD * abs(data[10]) + yy1), 1)
                cv2.circle(BearingImage, (int((data[1]) + 10 * math.sin(math.radians(data[11]))), int((data[2]) + 10 * math.cos(math.radians(data[11])))), 1, (mPD * abs(data[10]) + yy1, 0, 0), 1)
                # Displacement
                cv2.putText(BearingImage, "%.1f" % data[10], (int(data[1]) - 14, int(data[2]) - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, mPD * abs(data[10]) + yy1, 0), 1)
                # Bearing
                # cv2.putText(BearingImage, "%.2f" % data[12], (int(data[1]) - 14, int(data[2]) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, mPS * abs(data[12]) + yy1), 1)
                # Number
                cv2.putText(BearingImage, "%d" % data[0], (int(data[1]) - 14, int(data[2]) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150 - (mPD * abs(data[10]) + yy1), 150 - (mPD * abs(data[10]) + yy1), 150), 1)
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
            # Display the number of blobs.
            cv2.putText(frame_with_box, "Tracking %d pins" % DATA[-1][0], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            BlackImage[y1:y2, 0:x2-x1] = BearingImage
            video                      = np.concatenate((BlackImage, frame_with_box), axis=1)

            VectorField = copy.deepcopy(BearingImage)
            # Adding some black space to push the pins away from the edge of the frame
            blackBar    = np.zeros((150, VectorField.shape[1], 3), np.uint8)
            greybar     = np.zeros((2, VectorField.shape[1], 3), np.uint8); greybar.fill(50)
            VectorField = np.concatenate((VectorField,greybar, blackBar), axis=0)

            # Draw reference Blob
            fontsize = 0.4; Separation = 30
            BlobCoords = (VectorField.shape[1]-100, VectorField.shape[0]-75)
            cv2.circle(VectorField, BlobCoords, 4, (255,255,255), -1)
            cv2.line(VectorField, BlobCoords, (BlobCoords[0]+20, BlobCoords[1]-20), (255, 255, 255), 2)

            DisplSize = cv2.getTextSize("Displacement", cv2.FONT_HERSHEY_SIMPLEX, fontsize, 1)
            cv2.putText(VectorField, "Displacement", (BlobCoords[0] - (DisplSize[0][0]/2), BlobCoords[1] - DisplSize[0][1] - Separation), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (255,255,255))
            PinNumSize = cv2.getTextSize("Papillae Pin Number", cv2.FONT_HERSHEY_SIMPLEX, fontsize, 1)
            cv2.putText(VectorField, "Papillae Pin Number", (BlobCoords[0] - (PinNumSize[0][0]/2), BlobCoords[1] + PinNumSize[0][1] + Separation), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (255,255,255))
            # Show the frames
            # cv2.imshow("Camera", video)
            # cv2.imshow('frame2', ROI2)
            # cv2.imshow("Overview", frame_with_box)
            cv2.imshow("Bearings", VectorField)
            # print VectorField.shape
            if recording:
                VideoFrame.write(VectorField)
        except IndexError:
            initialize = 1

    if not ret:
        print "Camera not attached or in use."
        break

    k = cv2.waitKey(5) & 0xFF
    if k == 32:
        initialize = 1
    if k == 9:
        superimposeLines = ~superimposeLines
    if k == 114:
        print "Key pressed"
        if recording:
            print "Stopped recording"
            recording = 0
            VideoFrame.release()
        else:
            record = 1
    elif k == 27:
        if recording:
            VideoFrame.release()
        cam.release()
        cv2.destroyAllWindows()
        break
