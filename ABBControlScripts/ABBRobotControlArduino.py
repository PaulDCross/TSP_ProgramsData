from collections import OrderedDict
from datetime import datetime
import transformations as t
import numpy as np
import threading
import socket
import serial
import Queue
import math
import time
import copy
import cv2
import abb
import sys
import os

np.set_printoptions(precision=3, suppress=True)
connected       = threading.Event()
checkData       = threading.Event()
is_runningEvent = threading.Event()
q1Lock          = threading.Lock()
Newtons_lock    = threading.Lock()
robotLock       = threading.Lock()
pictureLock     = threading.Lock()
q1              = Queue.LifoQueue()
dic             = Queue.LifoQueue()
RobotQueue      = Queue.LifoQueue()
PictureQueue    = Queue.LifoQueue()

class ABBProgram:
    """docstring for ABBProgram"""
    def __init__(self):
        degree                  = "*"
        unit                    = "mm"
        self.distance           = 10
        self.WeightSamples      = 100
        self.WeightColums       = []
        self.height, self.Step, IP, HH, Tool = self.initialize()
        self.homePos            = [0.89, 9.13, -21.95, -0.31, 82.79, 0.88]
        self.homePosCart        = [[385.0, 5.0, 460.0], [0.173956, 0.0, 0.984753, 0.0]]
        self.HoverPosition      = [385.0, 5.0, HH, 180.0, 20.0, 180.0]
        self.movType            = ["X", "Y", "Z", "Rx", "Ry", "Rz"]
        dictionary              = OrderedDict()
        dictionary["cartesian"] = OrderedDict()
        d                       = dictionary["cartesian"]
        self.error              = 5
        # Make dictionary.
        [self.get_coordinates(d, i, self.distance, self.Step, copy.deepcopy(self.HoverPosition)) for i in xrange(6)]
        dic.put(d)
        print([len(d[i]) for i in self.movType])
        print "Hover Position: ", self.HoverPosition
        # Do you wish to start the experment of do you wish to abort now.
        while True:
            self.Start = raw_input('Start sequence? y/n: ')
            if self.Start in ("y", "n"):
                break
            else:
                print "Incorrect input, please try again.\n"
        # Convert self.HoverPosition from euler to quaternion.
        X                  = copy.deepcopy(self.HoverPosition)
        self.HoverPosition = self.d2q(X, 5)
        self.HoverPosition = [self.HoverPosition[:3], self.HoverPosition[3:]]

        if self.Start == "y":
            DIR = os.path.join("TSP_Pictures", "ArduinoWeightTest%.1f" % Tool[0][2], "%.1fmm" % self.height)
            self.makedir(DIR)
            self.numFolders = int(1 + len([name for name in os.listdir(DIR) if os.path.isdir(os.path.join(DIR, name))]))
            self.DIR        = os.path.join(DIR, "%02d" % self.numFolders)
            self.makedir(self.DIR)
            # Create the log file and open it.
            LogFile = os.path.join(self.DIR, "LogFile") + ".txt"
            self.f  = open(LogFile, 'w')
            self.startclock = time.time()
            self.printDateTime()
            print >> self.f, "Running program on Robot with IP address: ", IP,
            self.printDateTime()
            print >> self.f, "With tool: ", Tool
            self.Home()

    def initialize(self):
        # Choose the spacing in mm between each waypoint.
        while True:
            try:
                step = float(raw_input("Choose a step that is in the range of 0.1mm and 5mm: "))
                step = round(step, 1)
                if 0.1 <= step <= 10:
                    print "Accepted - The Robot will travel %dmm, with a step of %.1fmm.\n" % (self.distance, step)
                    break
            except:
                pass
            print "Incorrect input, please try again.\n"
        # Type the IP of the ABB robot that you wish to use. 127.0.0.1 for simulation and 164.11.72.*225 for actual robot.
        # *check with the robot this is correct.
        while True:
            try:
                IP     = raw_input('What is the IP of the Robot Controller: ')
                print "Connecting..."
                self.R = abb.Robot(ip=IP)
                # R    = abb.Robot(ip='164.11.72.125')
                print "Connected."
                # time.sleep(2)
                break
            except (socket.error, socket.timeout):
                pass
            print "Incorrect IP, please try again.\n"
        Tool = self.R.get_tool()
        if Tool[0][2] == 130:
            HH = "390.0"
        elif Tool[0][2] == 142.5:
            HH = "380.0"
        elif Tool[0][2] == 155:
            HH = "370.0"
        elif Tool[0][2] == 167.5:
            HH = 410.0
        elif Tool[0][2] == 180:
            HH = "340.0"
        else:
            HH = 460
        print "Tool: ", Tool
        # Choose the height that you wish to run at.
        while True:
            try:
                height = float(raw_input("What is the Z value e.g. {0} is just above the Pillow: ".format(HH)))
                if 330 <= height <= 450:
                    print "Accepted - The Robot will perform the run at a height of %.1fmm.\n" % height
                    break
            except:
                pass
            print 'Incorrect input, please try again.\n'
        return height, step, IP, HH, Tool

    def get_coordinates(self, d, i, distance, step, touchDownPos):
        XMovement = []
        touchDownPos[2] = self.height
        for newValue in self.linspace(touchDownPos[i], touchDownPos[i] - distance, step, "-"):
            # print newValue
            X    = copy.deepcopy(touchDownPos)
            X[i] = round(newValue, 1)
            XMovement.append(self.d2q(X, 5))
        for newValue in self.linspace(touchDownPos[i] - distance, touchDownPos[i], step, "+"):
            # print newValue
            X    = copy.deepcopy(touchDownPos)
            X[i] = round(newValue, 1)
            XMovement.append(self.d2q(X, 5))
        d[self.movType[i]] = XMovement
        return d

    def linspace(self, start, end, step, type):
        array = []
        temp = start
        if type == "+":
            while round(temp,1) <= round(end,1):
                array.append(round(temp,1))
                temp += step
        if type == "-":
            while round(temp,1) >= round(end,1):
                array.append(round(temp,1))
                temp -= step
        return array

    def d2r(self, degrees):
        degrees[3:6] = [(degrees[i] * math.pi) / 180 for i in range(3, 6)]
        return degrees

    def r2d(self, radians):
        degrees = [round((radians[i] * 180) / math.pi, 1) for i in range(3)]
        return degrees

    def d2q(self, X, rndFactor):
        X    = self.d2r(X)
        q    = t.quaternion_from_euler(X[3], X[4], X[5], 'sxyz')
        X[3] = round(q[0], rndFactor)
        X[4] = round(q[1], rndFactor)
        X[5] = round(q[2], rndFactor)
        X.append(round(q[3], rndFactor))
        return X

    def makedir(self, DIR):
        if not os.path.exists(DIR):
            os.makedirs(DIR)
            time.sleep(0.5)

    def printDateTime(self):
        print >> self.f, "\n", datetime.now().strftime('%d-%m-%Y %H:%M:%S'), "\t%.5f" % (time.time() - self.startclock), "\t\t",

    def Home(self):
        with robotLock:
            self.R.set_speed(speed=[10, 10, 50, 50])
            print "Speeding up to 10%"
            self.printDateTime()
            print >> self.f, "Heading to Home Position",
            self.f.flush()
            while (self.R.get_joints() != self.homePos):
                self.R.set_joints(self.homePos)
                time.sleep(4)
            self.R.set_speed(speed=[0.5, 0.5, 50, 50])
            print "Slowing Down to 1%"
            cartesian    = self.R.get_cartesian()
            euler        = self.r2d(t.euler_from_quaternion(cartesian[1]))
            euler        = [round(euler[i], 1) for i in range(len(euler))]
            cartesian[1] = euler
            self.printDateTime()
            print >> self.f, "Arrived at Home Position: ", cartesian
            self.f.flush()

    def hover(self):
        with robotLock:
            self.R.set_speed(speed=[10, 10, 50, 50])
            print "Speeding up to 10% 'hover'"
            self.printDateTime()
            print >> self.f, "Heading to Z coordinate %.1fmm" % self.HoverPosition[0][2],
            print "Heading to Z coordinate %.1fmm" % self.HoverPosition[0][2]
            self.f.flush()
            count = 0
            while (self.R.get_cartesian() != self.HoverPosition):
                self.R.set_cartesian(self.HoverPosition)
                time.sleep(1)
                a = self.R.get_cartesian()
                print a, self.HoverPosition
                if count >= 3:
                    self.R.set_speed(speed=[0.5, 0.5, 50, 50])
                count += 1
            self.R.set_speed(speed=[0.5, 0.5, 50, 50])
            print "Slowing Down to 1% 'hover'"
            cartesian    = self.R.get_cartesian()
            euler        = self.r2d(t.euler_from_quaternion(cartesian[1]))
            euler        = [round(euler[i], 1) for i in range(len(euler))]
            cartesian[1] = euler
            self.printDateTime()
            print >> self.f, "Hovering at: ", cartesian
            self.f.flush()
            time.sleep(0.2)
            saveImage()

    def runMain(self, units, i):
        if is_runningEvent.wait(0):
            with Newtons_lock:
                d = dic.get()
                dic.task_done()
                dic.put(d)
                data = d[self.movType[i]]
                DIR  = os.path.join(self.DIR, self.movType[i])
                self.makedir(DIR)
                self.directory = Queue.LifoQueue()
                self.directory.put(DIR)

            self.printDateTime()
            print >> self.f, "Starting Sequence: %s 0%s --> -20%s" % (DIR, units, units)
            self.hover()
            time.sleep(0.5)

            for dataline in data[:len(data) / 2]:
                if is_runningEvent.wait(0):
                    dataline = [dataline[:3], dataline[3:]]
                    euler    = copy.deepcopy(dataline)
                    euler[1] = self.r2d(t.euler_from_quaternion(euler[1]))
                    euler[1] = [round(euler[1][i], 1) for i in range(len(euler[1]))]
                    self.printDateTime()
                    print >> self.f, "Target: ", euler,
                    print "Starting Movement"
                    with robotLock:
                        while ([self.R.get_cartesian()[0], self.r2d(t.euler_from_quaternion(self.R.get_cartesian()[1]))] != euler) and is_runningEvent.wait(0): # [dataline[0], [round(line,3) for line in dataline[1]]]
                            self.R.set_cartesian(dataline)
                            time.sleep(0.5)
                            a = self.R.get_cartesian()
                            b = [self.R.get_cartesian()[0], self.r2d(t.euler_from_quaternion(self.R.get_cartesian()[1]))]
                            print a, dataline # [dataline[0], [round(line,3) for line in dataline[1]]]
                            print b, euler
                            print "Equal? ", [[abs(c1) for c1 in b[0]], [abs(c2) for c2 in b[1]]] == [[abs(e1) for e1 in euler[0]], [abs(e2) for e2 in euler[1]]]
                            if [[abs(c1) for c1 in b[0]], [abs(c2) for c2 in b[1]]] == [[abs(e1) for e1 in euler[0]], [abs(e2) for e2 in euler[1]]]:
                                break
                        euler    = self.R.get_cartesian()
                    euler[1] = self.r2d(t.euler_from_quaternion(euler[1]))
                    euler[1] = [round(euler[1][i], 1) for i in range(len(euler[1]))]
                    self.printDateTime()
                    print >> self.f, "Result: ", euler,
                    self.f.flush()
                    print "Result: ", euler
                    print "Finished Movement"
                    self.get300samples(self.WeightSamples, euler[0][2])
                    time.sleep(0.2)
                    saveImage()

            time.sleep(1)
            for dataline in data[len(data) / 2 : len(data)]:
                if is_runningEvent.wait(0):
                    dataline = [dataline[:3], dataline[3:]]
                    euler    = copy.deepcopy(dataline)
                    euler[1] = self.r2d(t.euler_from_quaternion(euler[1]))
                    euler[1] = [round(euler[1][i], 1) for i in range(len(euler[1]))]
                    self.printDateTime()
                    print >> self.f, "Target: ", euler,
                    print "Starting Movement"
                    with robotLock:
                        while ([self.R.get_cartesian()[0], self.r2d(t.euler_from_quaternion(self.R.get_cartesian()[1]))] != euler) and is_runningEvent.wait(0): # [dataline[0], [round(line,3) for line in dataline[1]]]
                            self.R.set_cartesian(dataline)
                            time.sleep(0.5)
                            a = self.R.get_cartesian()
                            b = [self.R.get_cartesian()[0], self.r2d(t.euler_from_quaternion(self.R.get_cartesian()[1]))]
                            print a, dataline # [dataline[0], [round(line,3) for line in dataline[1]]]
                            print b, euler
                            print "Equal? ", [[abs(c1) for c1 in b[0]], [abs(c2) for c2 in b[1]]] == [[abs(e1) for e1 in euler[0]], [abs(e2) for e2 in euler[1]]]
                            if [[abs(c1) for c1 in b[0]], [abs(c2) for c2 in b[1]]] == [[abs(e1) for e1 in euler[0]], [abs(e2) for e2 in euler[1]]]:
                                break
                        euler    = self.R.get_cartesian()
                    euler[1] = self.r2d(t.euler_from_quaternion(euler[1]))
                    euler[1] = [round(euler[1][i], 1) for i in range(len(euler[1]))]
                    self.printDateTime()
                    print >> self.f, "Result: ", euler,
                    self.f.flush()
                    print "Result: ", euler
                    print "Finished Movement"
                    self.get300samples(self.WeightSamples, euler[0][2])
                    time.sleep(0.2)
                    saveImage()

    def get300samples(self, samples, Height):
        column = []
        appendc = column.append
        i = 0
        print "Gathering samples..."
        while i <= samples and is_runningEvent.wait(0):
            with q1Lock:
                WeightArray = q1.get()
                q1.task_done()
            time.sleep(0.1)
            WeightArray.append(Height)
            appendc(WeightArray)
            # print WeightArray
            i += 1
        self.WeightColums.append(column)
        print "Gathered {0} samples.".format(i-1)

    def close(self):
        print "Goodbye..."
        checkData.clear()
        is_runningEvent.clear()
        time.sleep(1)
        self.R.close()

def writeList2File(textFile, DATA):
    with open(textFile, "w") as file:
        DATA = '\n'.join('\t\t'.join(map(str,j)) for j in DATA).replace(', ', '\t').replace('[', '').replace(']', '')
        file.write(str(DATA))

def saveImage():#, image):
    t0 = time.time()
    with Newtons_lock:
        Robot = RobotQueue.get()
        DIR   = Robot.directory.get()
        RobotQueue.task_done()
        RobotQueue.put(Robot)
        Robot.directory.task_done()
        Robot.directory.put(DIR)
    with pictureLock:
        frame = PictureQueue.get()
        PictureQueue.task_done()

    dirc = os.path.join(DIR, "Internal")
    Robot.makedir(dirc)
    name = int(1 + len([name for name in os.listdir(dirc) if os.path.isfile(os.path.join(dirc, name))]))
    name = os.path.join(dirc, "%003d" % name) + ".png"
    cv2.imwrite(name, frame)
    Robot.printDateTime()
    print >> Robot.f, "Picture saved:", name
    Robot.f.flush()
    print "Picture saved:", name
    t1 = time.time()
    print t1-t0


def takeImage():
    cam1 = cv2.VideoCapture(0)   # Start the webcam
    cam1.set(3, 1200)            # horizontal pixels
    cam1.set(4, 720)             # vertical pixels
    print "Cameras Opened? ", cam1.isOpened()
    while cam1.isOpened() and is_runningEvent.wait(0):# and cam2.isOpened():
        _, frame = cam1.read()
        time.sleep(0.5)
        if frame.any():
            with pictureLock:
                PictureQueue.put(frame)
    cam1.release()
    cv2.destroyAllWindows()

def Scale():
    comPort  = "com4"
    baudrate = 9600
    width    = 280
    height   = 140
    white    = (255, 255, 255)
    black    = (0, 0, 0)
    red      = (0, 0, 225)
    green    = (0, 225, 0)
    try:
        print("Connecting to Arduino...")
        arduinoSerialData = serial.Serial(comPort, baudrate)
        print("Connected to com4 with baud rate of 9600.")
        time.sleep(2)
        connected.set()
        is_runningEvent.set()
    except serial.serialutil.SerialException:
        print("Connection failed.")
        print("Exiting thread...")
        is_runningEvent.clear()
    while is_runningEvent.wait(0):
        try:
            array          = [float(value) for value in arduinoSerialData.readline().strip().split(",")]
            with q1Lock:
                q1.put(array)
            checkData.set()
            UI             = np.zeros([height, width, 3])
            UI[:, :]       = white
            UI[:, 139:141] = black
            update         = cv2.getTextSize(str(array[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
            update2        = cv2.getTextSize(str(array[2]), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
            cv2.putText(UI, str(array[1]), ((width/4) - (update[0][0]/2), (height/2) + (update[0][1] / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
            cv2.putText(UI, str(array[2]), (int(width/1.333) - (update2[0][0]/2), (height/2) + (update2[0][1] / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
            cv2.imshow("UI", UI)
            cv2.moveWindow("UI", 50, 50)
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyAllWindows()
                is_runningEvent.clear()
        except(ValueError, IndexError):
            pass

def main():
    degree = "*"
    unit   = "mm"

    Robot  = ABBProgram()
    RobotQueue.put(Robot)

    if Robot.Start == "y":
        ScaleThread        = threading.Thread(target=Scale, name="Scales")
        ScaleThread.daemon = True
        ScaleThread.start()

        connected.wait(0.2)
        if is_runningEvent.wait(3):
            print "Scale is connected and running."

            TakePicture        = threading.Thread(target=takeImage, name="Take Picture")
            TakePicture.daemon = True
            TakePicture.start()

            if not checkData.wait(5):
                print "No data received... Shutting Down."
                Robot.close()
                time.sleep(2)
                print threading.enumerate()
                sys.exit(0)
            print "Receiving Data."

            [Robot.runMain(unit, i) for i in xrange(2, 3)]
            # [Robot.runMain(degree, i) for i in xrange(3, 5)]
            print "Saving WeightLogFile"
            DATA1 = ['\t'.join(map(str, b)) for a in Robot.WeightColums[: len(Robot.WeightColums)/2] for b in a]
            DATA2 = ['\t'.join(map(str, b)) + '\n' for a in reversed(Robot.WeightColums[len(Robot.WeightColums)/2 :]) for b in a]
            for data1, data2 in zip(DATA1, DATA2):
                with open(os.path.join(Robot.DIR, "WeightLogFile.txt"), 'a') as file:
                    dat = data1 + '\t\t' + data2
                    file.write(dat)
            np.save(os.path.join(Robot.DIR, "WeightHeightData"), np.array(Robot.WeightColums))

            Robot.Home()
            Robot.printDateTime()
            print >> Robot.f, "Program Finished."
    time.sleep(0.5)
    raw_input('Close program?')
    Robot.close()
    time.sleep(2)
    print threading.enumerate()

thread = threading.Thread(target=main, name="Main")
thread.start()