import cv2
from operator import itemgetter
import math
import itertools
import numpy as np
from decimal import Decimal
import os

__all__ = ['chunker', 'Pillow', 'rw', 'click_and_crop', 'Pins']

def chunker(seq, size):
    return [seq[pos:pos + size] for pos in xrange(0, len(seq), size)]

class Pillow:

    def __init__(self, frame, refPt):
        self.frame = frame
        self.refPt = refPt

    def getFrame(self):
        """ Get the frame from the camera, convert it to gray, threshold it,
        apply some transformations, then crop it"""

        x1,y1 = self.refPt[0][0],self.refPt[0][1]
        x2,y2 = self.refPt[1][0],self.refPt[1][1]
        # Crop to ROI
        frame = self.frame[y1:y2, x1:x2]

        # Convert the frame to GRAY, and blur it
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresholded = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)[1]
        # blur = cv2.GaussianBlur(image, (5,5), 0)
        # ret, thresholded = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) #200 255

        # Dialate the thresholded image to fill in holes
        image = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, None, iterations = 1)
        ROI = cv2.dilate(image, None, iterations = 4)

        frame_with_box = cv2.rectangle(self.frame, (x1,y1), (x2,y2), (0,255,0), 1)

        return ROI, frame_with_box



    def detectorParameters(self):
        """Set up the blob detector parameters"""
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 150;
        params.maxThreshold = 255;
        # Filter by Colour
        params.filterByColor = True
        params.blobColor = 255
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 90 # 120, 142
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.7 # 0.8
        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 4:
            self.detector = cv2.SimpleBlobDetector_create(params)
        else :
            self.detector = cv2.SimpleBlobDetector_create(params)
        return self.detector


    def measurements(self,data1,data2,L2):
        """get the measurements"""
        self.DistanceBearing, dy, dx = [], [], []

        for i in xrange(L2):
            # Calculate the distance traveled
            dy.append((data2[i][2] - data1[i][2]))
            dx.append((data2[i][1] - data1[i][1]))
            distance = round(math.sqrt((round(dx[i], 1))**2 + (round(dy[i], 1))**2), 1)
            changeinSize = round(data2[i][3] - data1[i][3],2)

            # Calculate the bearing
            if dx[i] > 0:
                if dy[i] > 0:
                    bearing = math.atan((dx[i]/dy[i]))
                elif dy[i] < 0:
                    bearing = math.pi + math.atan((dx[i]/dy[i]))
                else:
                    bearing = (math.pi)/2
            elif dx[i] < 0:
                if dy[i] > 0:
                    bearing = 2*(math.pi) + math.atan((dx[i]/dy[i]))
                elif dy[i] < 0:
                    bearing = math.pi + math.atan((dx[i]/dy[i]))
                else:
                    bearing = (3/2)*math.pi
            else:
                if dy[i] > 0:
                    bearing = 0
                elif dy[i] < 0:
                    bearing = math.pi
                else:
                    bearing = 0

            if (distance < 2.5 or distance > 25):
                state = 0.0
            else:
                state = 1.0

            bearing = round(bearing*(180/math.pi),1)
            # distance = round(math.pow(distance,2),1)
            self.DistanceBearing.append([data1[i][0], state, dx[i], dy[i], distance, bearing,changeinSize])
            self.DistanceBearing.sort(key = itemgetter(0),reverse = False)

            if len(self.DistanceBearing) == L2:
                from Pillow import rw
                rw().writeList2File("Pin_Distances_Bearings.txt", self.DistanceBearing)
        return self.DistanceBearing


    def blobCheck(self, coords, xyn):
        """ get the blob name from a predefined blobposition"""
        pins = Pins(self.refPt)
        Rows, _ = pins.countRnC(xyn)
        for i in xrange(len(xyn)):
            if coords[1] < xyn[i][1]:
                for j in xrange(Rows):
                    if coords[0] < xyn[j][0]:
                        return xyn[i + j][2]


    def initialiseData(self,xyn):
        """Setup the frame for blob detection"""
        ROI, _ = self.getFrame()
        # Set the detectors parametors and detect blobs.
        coords1 = [x.pt + (x.size,) for x in self.detectorParameters().detect(ROI)]
        data1 = []

        for coordinates in coords1:
            coords = [round(coordinates[0],1), round(coordinates[1],1)]
            blobNum1 = self.blobCheck(coords,xyn)
            data1.append([int(blobNum1),round(coordinates[0],1), round(coordinates[1],1),round(coordinates[2],3)])

        data1.sort(key=itemgetter(0), reverse = False)
        return data1


    def getDataSet2(self, keypoints, xyn):
        """Get the second data set"""
        coords2 = [x.pt + (x.size,) for x in keypoints]
        data2 = []

        for coordinates in coords2:
            coords = [round(coordinates[0],1), round(coordinates[1],1)]
            blobNum1 = self.blobCheck(coords,xyn)
            data2.append([int(blobNum1),round(coordinates[0],1), round(coordinates[1],1),round(coordinates[2],3)])

        data2.sort(key=itemgetter(0), reverse = False)
        return data2

    def colorMap(self, DistanceBearing):
        """map the blob data with a certain colour"""
        colorMap = []

        for i in xrange(len(DistanceBearing)):
            Colour = int(100 + math.pow((DistanceBearing[i][2]), 2))

            # if DistanceBearing[i][3] <= 120:
            #   Colour = (int(np.interp(DistanceBearing[i][3], [0,120], [50,255]))) + (DistanceBearing[i][2] * 10)
            #   color = (0, Colour, 0)
            # elif DistanceBearing[i][3] <= 240:
            #   Colour = (int(np.interp(DistanceBearing[i][3], [121,240], [50,255]))) + (DistanceBearing[i][2] * 10)
            #   color = (0, 0, Colour)
            # elif DistanceBearing[i][3] <= 360:
            #   Colour = (int(np.interp(DistanceBearing[i][3], [241,360], [50,255]))) + (DistanceBearing[i][2] * 10)
            #   color = (Colour, 0, 0)

            if (DistanceBearing[i][2] < 10):
                color = (Colour, 0, 0)
            elif DistanceBearing[i][2] < 15:
                color = (Colour, Colour, 0)
            elif DistanceBearing[i][2] < 20:
                color = (0, Colour, 0)
            elif DistanceBearing[i][2] < 30:
                color = (0, Colour, Colour)
            elif DistanceBearing[i][2] < 40:
                color = (0, 0, Colour)

            colorMap.append(color)
            # print colorMap
        return colorMap



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


class click_and_crop():
    def crop_main(self, image):
        # initialize the list of reference points and boolean indicating
        # whether cropping is being performed or not
        self.refPt = []
        self.cropping = False

        self.image = image
        # Clone the image and setup the mouse callback function
        self.clone = self.image.copy()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.click_and_crop)

        # keep looping until the "c" key is pressed
        while True:
            # display the image and wait for a keypress
            cv2.imshow("image", self.image)
            key = cv2.waitKey(1) & 0xFF

            # if the "r" key is pressed, reset the cropping region
            if key == ord('r'):
                self.image = self.clone.copy()

            # if the "c" key is pressed, break from the loop
            elif key == ord("c"):
                break

        # if there are two reference points, then crop the region of interest
        # from the image and display it
        if len(self.refPt) == 2:
            roi = self.clone[self.refPt[0][1]:self.refPt[1][1], self.refPt[0][0]:self.refPt[1][0]]

        # close all open windows
        cv2.destroyAllWindows()
        print self.refPt
        return self.refPt

    def click_and_crop(self, event, x, y, flags, param):
        # if the left mouse buttin was clicked, record the starting
        # (x,y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            self.refPt = [(x,y)]
            self.cropping = True

        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x,y) coordinates and indicate that
            # the cropping ooperation is finished.
            self.refPt.append((x,y))
            self.cropping = False

            # draw a rectangle around the region of interest
            cv2.rectangle(self.image, self.refPt[0],self.refPt[1], (0,255,0), 2)
            cv2.imshow("image", self.image)
            return self.image

    # Write this:
    # cc = click_and_crop()
    # refPt = cc.crop_main(frame)


class Pins(): # Cuts up the image of pins
    def __init__(self,refPt):
        self.x1, self.y1 = refPt[0][0],refPt[0][1]
        self.x2, self.y2 = refPt[1][0],refPt[1][1]

    def countRnC(self,data):
        Columns = 1
        Rows = 1
        gap = 10
        for i in xrange(len(data)-1):
            if ((data[i][1] + gap) > data[i+1][1]) & (data[i+1][1] > (data[i][1] - gap)):
                Columns = Columns + 1
            else:
                Rows = Rows + 1
                Columns = 1
        return Columns, Rows

    def vertical(self,Rows,coords):
        coords = sorted(coords, key=itemgetter(0))
        b, minmax = [], []
        for a in xrange(0,len(coords),Rows):
            b.append(coords[a:a+Rows])
        for i in xrange(len(b)):
            c = []
            for j in xrange(len(b[i])):
                c.append(b[i][j][0])
            minmax.append([min(c),max(c)])

        minmax[len(minmax)-1][1] = self.x2-self.x1
        minmax.append([self.x2-self.x1,self.x2-self.x1])
        midpoints = [(round((minmax[i+1][0] - minmax[i][1]),2))/2 for i in xrange(len(minmax)-1)]
        return midpoints, minmax

    def horizontal(self,Columns,coords):
        b, minmax = [], []
        for a in xrange(0,len(coords),Columns):
            b.append(coords[a:a+Columns])
        for i in xrange(len(b)):
            c = []
            for j in xrange(len(b[i])):
                c.append(b[i][j][1])
            minmax.append([min(c),max(c)])

        minmax[len(minmax)-1][1] = self.y2-self.y1
        minmax.append([self.y2-self.y1,self.y2-self.y1])
        midpoints = [(round((minmax[i+1][0] - minmax[i][1]),2))/2 for i in xrange(len(minmax)-1)]
        return midpoints, minmax

    def main(self,keypoints):
        roundedCoordinates, crosspointsx, crosspointsy = [], [], []
        coordinates = [keypoints[i-1].pt for i in xrange(len(keypoints))]
        coordinates = sorted(coordinates, key=itemgetter(1))

        [roundedCoordinates.append((round(coordinates[i][0],1),round(coordinates[i][1],1))) for i in xrange(len(coordinates))]

        Columns, Rows = self.countRnC(roundedCoordinates)
        Vmidpoints, Vminmax = self.vertical(Rows,roundedCoordinates)
        Hmidpoints, Hminmax = self.horizontal(Columns,roundedCoordinates)

        for i in xrange(len(Hmidpoints)):
            for j in xrange(len(Vmidpoints)):
                crosspointsx.append((int(Vmidpoints[j])+int(Vminmax[j][1])))
                crosspointsy.append((int(Hmidpoints[i])+int(Hminmax[i][1])))
        crosspoints = zip(crosspointsx,crosspointsy)

        for i in xrange(len(crosspoints)):
            crosspoints[i] = crosspoints[i]+(i+1,)
        rw().writeList2File("Pin_Regions.txt",crosspoints)
        return Columns, Rows, crosspoints


"""
        w = 600
        h = 300

        sw = self.parent.winfo_screenwidth()
        sh = self.parent.winfo_screenheight()

        x = (sw-w)/2
        y = (sh-h)/2
        self.parent.geometry("%dx%d+%d+%d" % (w,h,x,y))
"""

