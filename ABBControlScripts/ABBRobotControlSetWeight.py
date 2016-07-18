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
Newtons_lock     = threading.Lock()
Newtons_lock2    = threading.Lock()
touchDownPosLock = threading.Lock()
robotLock        = threading.Lock()
print_lock       = threading.Lock()
connected        = threading.Event()
checkData        = threading.Event()
StartEvent       = threading.Event()
saveImgEvent     = threading.Event()
isTouchedDown    = threading.Event()
Z                = threading.Event()
GUIupdate        = threading.Event()
end              = threading.Event()
initErrorBds     = threading.Event()
q1               = Queue.LifoQueue()
Z1               = Queue.LifoQueue()
Z2               = Queue.LifoQueue()
KG               = Queue.LifoQueue()
dic              = Queue.LifoQueue()
touchDownQueue   = Queue.LifoQueue()
RobotQueue       = Queue.LifoQueue()


class ABBProgram:
    """docstring for ABBProgram"""
    def __init__(self):
        degree                           = "*"
        unit                             = "mm"
        distance                         = 20
        self.homePos                     = [0.89, 10.77, -26.32, -0.30, 85.55, 0.86]
        self.homePosCart                 = [[390, 5, 460], [0.174, 0, 0.985, 0]]
        self.HoverPosition               = [390.0, 5, 392.5, 180.0, 20.0, 180.0]
        self.movType                     = ["X", "Y", "Z", "Rx", "Ry", "Rz"]
        dictionary                       = OrderedDict()
        dictionary["cartesian"]          = OrderedDict()
        d                                = dictionary["cartesian"]
        self.error                       = 5
        self.kg, N, self.Step, IP, Start = self.initialize()
        KG.put(self.kg)
        # Make dictionary.
        [self.get_coordinates(d, i, distance, self.Step, self.HoverPosition) for i in xrange(6)]
        dic.put(d)
        print([len(d[i]) for i in self.movType])
        # Convert self.HoverPosition from euler to quaternion.
        X                 = copy.copy(self.HoverPosition)
        self.HoverPosition = self.d2q(X, 3)
        self.HoverPosition = [self.HoverPosition[:3], self.HoverPosition[3:]]
        if Start == "y":
            DIR = os.path.join("TSP_Pictures", "%dgrams" % self.kg)
            self.makedir(DIR)
            self.numFolders = int(1 + len([name for name in os.listdir(DIR) if os.path.isdir(os.path.join(DIR, name))]))
            self.DIR        = os.path.join(DIR, "%02d" % self.numFolders)
            self.makedir(self.DIR)
            # Create the log file and open it.
            LogFile = os.path.join(self.DIR, datetime.now().strftime('%Y%m%d_%H_%M_%S')) + ".txt"
            self.f  = open(LogFile, 'w')
            StartEvent.set()
            self.startclock = time.time()
            self.printDateTime()
            print >> self.f, "Running program on Robot with IP address: ", IP
            self.Home()
        else:
            StartEvent.clear()

    def initialize(self):
        # Choose the weight that you wish to simulate.
        while True:
            try:
                kg = float(raw_input("Choose a multiple of 0.1 in the range of 0.1kg to 1kg?: "))
                if 0.1 <= kg <= 1:
                    N = kg * 9.8
                    kg = kg*1000
                    print "Accepted - The Robot will simulate a head of %.1fgrams.\n" % kg
                    break
            except:
                pass
            print 'Incorrect input, please try again.\n'
        # Choose the spacing in mm between each waypoint.
        while True:
            try:
                step = float(raw_input("Choose a step that is in the range of 0.10mm and 5mm: "))
                if 0.1 <= step <= 10:
                    step = round(step, 1)
                    print "Accepted - The Robot will travel 20mm, with a step of %.1fmm.\n" % step
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
        # Do you wish to start the experment of do you wish to abort now.
        while True:
            Start = raw_input('Start sequence? y/n: ')
            if Start in ("y", "n"):
                break
            else:
                print "Incorrect input, please try again.\n"
        return kg, N, step, IP, Start

    def makedir(self, DIR):
        if not os.path.exists(DIR):
            os.makedirs(DIR)
            time.sleep(0.5)

    def printDateTime(self):
        print >> self.f, "\n", datetime.now().strftime('%d-%m-%Y %H:%M:%S'), "\t%.5f" % (time.time() - self.startclock), "\t\t",

    def d2r(self, degrees):
        degrees[3:6] = [(degrees[i] * math.pi) / 180 for i in range(3, 6)]
        return degrees

    def r2d(self, radians):
        radians = [round((radians[i] * 180) / math.pi, 3) for i in range(3)]
        return radians

    def d2q(self, X, rndFactor):
        X    = self.d2r(X)
        q    = t.quaternion_from_euler(X[3], X[4], X[5], 'sxyz')
        X[3] = round(q[0], rndFactor)
        X[4] = round(q[1], rndFactor)
        X[5] = round(q[2], rndFactor)
        X.append(round(q[3], rndFactor))
        return X

    def linspace(self, start, end, step, type):
        array = []
        temp = start
        if type == "+":
            while round(temp,1) <= round(end,1):
                array.append(temp)
                temp += step
        if type == "-":
            while round(temp,1) >= round(end,1):
                array.append(temp)
                temp -= step
        return array

    def get_coordinates(self, d, i, distance, step, touchDownPos):
        XMovement = []
        for newValue in self.linspace(touchDownPos[i], touchDownPos[i] + distance, step, "+"):
            # print newValue
            X    = copy.copy(touchDownPos)
            X[i] = round(newValue, 3)
            XMovement.append(self.d2q(X, 3))
        for newValue in self.linspace(touchDownPos[i], touchDownPos[i] - distance, step, "-"):
            # print newValue
            X    = copy.copy(touchDownPos)
            X[i] = round(newValue, 3)
            XMovement.append(self.d2q(X, 3))
        d[self.movType[i]] = XMovement
        return d

    def Home(self):
        with robotLock:
            initErrorBds.set()
            self.error = 5
            self.R.set_speed(speed=[10, 10, 50, 50])
            with print_lock:
                print "Speeding up to 10%"
            self.printDateTime()
            print >> self.f, "Heading to Home Position",
            self.f.flush()
            while (self.R.get_joints() != self.homePos):
                self.R.set_joints(self.homePos)
                time.sleep(4)
            self.R.set_speed(speed=[0.5, 0.5, 50, 50])
            with print_lock:
                print "Slowing Down to 1%"
            cartesian    = self.R.get_cartesian()
            euler        = self.r2d(list(t.euler_from_quaternion(cartesian[1])))
            euler        = [round(euler[i], 1) for i in range(len(euler))]
            cartesian[1] = euler
            self.printDateTime()
            print >> self.f, "Arrived at Home Position: ", cartesian
            self.f.flush()

    def hover(self):
        with robotLock:
            initErrorBds.set()
            self.error = 5
            self.R.set_speed(speed=[10, 10, 50, 50])
            with print_lock:
                print "Speeding up to 10% 'hover'"
            self.printDateTime()
            print >> self.f, "Heading to Z coordinate 345mm",
            self.f.flush()
            while (self.R.get_cartesian() != self.HoverPosition):
                self.R.set_cartesian(self.HoverPosition)
                time.sleep(1)
            self.R.set_speed(speed=[0.5, 0.5, 50, 50])
            with print_lock:
                print "Slowing Down to 1% 'hover'"
            cartesian    = self.R.get_cartesian()
            euler        = self.r2d(list(t.euler_from_quaternion(cartesian[1])))
            euler        = [round(euler[i], 1) for i in range(len(euler))]
            cartesian[1] = euler
            self.printDateTime()
            print >> self.f, "Hovering at: ", cartesian
            self.f.flush()
            return cartesian

    def getWeight(self):
        self.LoadCellArray = [("Time", "ADC", "Weight", "avgWeight", "cartesian[0][0]", "cartesian[0][1]", "cartesian[0][2]", "cartesian[1][0]", "cartesian[1][1]", "cartesian[1][2]", "cartesian[1][3]")]
        writeList2File(os.path.join("TSP_Pictures", "%dgrams" % self.kg, "%02d" % self.numFolders, "LoadCellData.txt"), self.LoadCellArray)
        while checkData.wait(0) and not end.wait(0):
            try:
                scaleArray = q1.get(True, 0.1)
                q1.task_done()
            except Queue.Empty:
                break
            if robotLock.acquire(False):
                cartesian   = self.R.get_cartesian()
                cartesian1  = copy.copy(cartesian)
                robotLock.release()
            else:
                cartesian   = copy.copy(cartesian1)
            touchDownQueue.put(cartesian)
            cartesian[1]       = self.r2d(list(t.euler_from_quaternion(cartesian[1], 'sxyz')))
            checktime          = time.time()
            self.LoadCellArray = ((round((checktime - self.startclock), 6), scaleArray[0], scaleArray[1], scaleArray[2], cartesian1[0][0], cartesian1[0][1], cartesian1[0][2], cartesian1[1][0], cartesian1[1][1], cartesian1[1][2], cartesian1[1][3]))
            writeList2File(os.path.join("TSP_Pictures", "%dgrams" % self.kg, "%02d" % self.numFolders, "LoadCellData.txt"), self.LoadCellArray)
            Z1.put(scaleArray)
            Z2.put(scaleArray[2])
            Z.set()
            GUIupdate.set()
            time.sleep(0.2)
        with print_lock:
            print "No Data1"

    def simulateWeight(self, Position):
        oldPos = Position
        cG = 0
        while not end.wait(0):
            Z.wait()
            try:
                avgWeight = Z2.get(True, 0.1)
            except Queue.Empty:
                break
            if avgWeight < self.kg - self.error:
                Position[0][2] = round(Position[0][2] - 0.1, 2)
                with print_lock:
                    print "DOWN"
                with robotLock:
                    while (self.R.get_cartesian() != Position) and not end.wait(0):
                        self.R.set_cartesian(Position)
                        time.sleep(1)
                        with print_lock:
                            a = self.R.get_cartesian()
                            print a, Position
                oldPos = Position
                isTouchedDown.clear()
                if Position[0][2] == 380:
                    with print_lock:
                        print "LOAD CELL ERROR - Min"
                    self.Home()
                    self.close()
                cG = 0
            elif avgWeight > self.kg + self.error:
                Position[0][2] = round(Position[0][2] + 0.1, 2)
                with print_lock:
                    print "UP"
                with robotLock:
                    while (self.R.get_cartesian() != Position) and not end.wait(0):
                        self.R.set_cartesian(Position)
                        time.sleep(1)
                        with print_lock:
                            a = self.R.get_cartesian()
                            print a, Position
                        if [abs(b) for b in a[1]] in [[0.172, 0.021, 0.978, 0.12], [0.311, 0.0, 0.95, 0.0]]:
                            break
                oldPos = Position
                isTouchedDown.clear()
                if Position[0][2] == 460:
                    with print_lock:
                        print "LOAD CELL ERROR - Max"
                    self.Home()
                    self.close()
                cG = 0
            else:
                with print_lock:
                    print "Touched Down"
                with robotLock:
                    euler    = copy.copy(Position)
                    euler[1] = self.r2d(list(t.euler_from_quaternion(euler[1])))
                    euler[1] = [round(euler[1][i], 1) for i in range(len(euler[1]))]
                    self.printDateTime()
                    print >> self.f, "Target: ", euler,
                    while (self.R.get_cartesian() != Position) and not end.wait(0):
                        self.R.set_cartesian(Position)
                        time.sleep(1)
                        with print_lock:
                            a = self.R.get_cartesian()
                            print a, Position
                        if [abs(b) for b in a[1]] in [[0, 0, 1, 0], [0.174, 0.004, 0.984, 0.024], [0.172, 0.021, 0.978, 0.12], [0.311, 0, 0.95, 0], [0.326, 0, 0.945, 0], [0.151, 0, 0.989, 0], [0.105, 0, 0.994, 0], [0.174, 0.024, 0.984, 0.004], [0.172, 0.12, 0.978, 0.021]]:#, [0.172, 0.12, 0.978, 0.021], [0.271, 0, 0.963, 0], [0.326, 0, 0.945, 0], [0.105, 0, 0.994, 0]]:
                            break
                    a = self.R.get_cartesian()
                    oldPos = Position
                    euler = copy.copy(a)
                    euler[1] = self.r2d(list(t.euler_from_quaternion(euler[1])))
                    euler[1] = [round(euler[1][i], 1) for i in range(len(euler[1]))]
                self.printDateTime()
                print >> self.f, "Result: ", euler,
                self.f.flush()
                with print_lock:
                    print euler
                cG += 1
                if cG == 2:
                    cG = 0
                    if initErrorBds.wait(0):
                        initErrorBds.clear()
                        self.error = 7.5
                    break
                isTouchedDown.set()
                Z.clear()
            time.sleep(2)
        return oldPos

    def runMain(self, units, i):
        if checkData.wait(0) or not end.wait(0):
            with Newtons_lock:
                d              = dic.get()
                dic.task_done()
                dic.put(d)
                data           = d[self.movType[i]]
                DIR            = os.path.join(self.DIR, self.movType[i], "P")
                self.makedir(DIR)
                self.directory = Queue.LifoQueue()
                self.directory.put(DIR)
            # print >> self.f, "\n############################################################################################################"
            self.printDateTime()
            print >> self.f, "Starting Sequence: %s 0%s --> +20%s" % (DIR, units, units)
            oldPos = self.hover()
            time.sleep(1)
            for dataline in data[:len(data) / 2]:
                dataline       = [dataline[:3], dataline[3:]]
                dataline[0][2] = oldPos[0][2]
                oldPos         = self.simulateWeight(dataline)
                if isTouchedDown.wait(0):
                    time.sleep(0.2)
                    saveImgEvent.clear()
                    if not saveImgEvent.wait(1):
                        break
            with Newtons_lock:
                DIR            = os.path.join(DIR[:-2], "N")
                self.makedir(DIR)
                self.directory = Queue.LifoQueue()
                self.directory.put(DIR)
            # print >> self.f, "\n############################################################################################################"
            self.printDateTime()
            print >> self.f, "Starting Sequence: %s 0%s --> -20%s" % (DIR, units, units)
            oldPos = self.hover()
            time.sleep(1)
            for dataline in data[len(data) / 2:len(data)]:
                dataline       = [dataline[:3], dataline[3:]]
                dataline[0][2] = oldPos[0][2]
                oldPos         = self.simulateWeight(dataline)
                if isTouchedDown.wait(0):
                    time.sleep(0.2)
                    saveImgEvent.clear()
                    if not saveImgEvent.wait(1):
                        break

    def close(self):
        with print_lock:
            print "Goodbye..."
        checkData.clear()
        time.sleep(1)
        self.R.close()
        end.set()


def saveImage(frame):
    with Newtons_lock:
        Robot = RobotQueue.get()
        DIR   = Robot.directory.get()
        RobotQueue.task_done()
        RobotQueue.put(Robot)
        Robot.directory.task_done()
        Robot.directory.put(DIR)
    name       = int(1 + len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))
    name       = os.path.join(DIR, "%003d" % name) + ".png"
    cv2.imwrite(name, frame)
    Robot.printDateTime()
    print >> Robot.f, "Picture saved:", name
    Robot.f.flush()
    with print_lock:
        print "Picture saved:", name
    saveImgEvent.set()


def takeImage():
    cam = cv2.VideoCapture(0)   # Start the webcam
    cam.set(3, 1200)            # horizontal pixels
    cam.set(4, 720)            # vertical pixels
    while cam.isOpened() and not end.wait(0):
        _, frame = cam.read()
        if not saveImgEvent.wait(0) and isTouchedDown.wait(0):
            saveImage(frame)
    cam.release()
    cv2.destroyAllWindows()


def Scale():
    comPort  = "com4"
    baudrate = 9600
    try:
        with print_lock:
            print("Connecting to Arduino...")
            arduinoSerialData = serial.Serial(comPort, baudrate)
            print("Connected to com4 with baud rate of 9600.")
            connected.set()
            time.sleep(2)
    except serial.serialutil.SerialException:
        with print_lock:
            print("Connection failed.")
            print("Exiting...")
            sys.exit(0)

    while not end.wait(0):
        try:
            array = [float(value) for value in arduinoSerialData.readline().strip().split(",")]
            q1.put(array)
            checkData.set()
        except(ValueError, IndexError):
            pass


def GUI():
    width   = 400
    height  = 400
    white   = (255, 255, 255)
    black   = (0, 0, 0)
    red     = (0, 0, 225)
    green   = (0, 225, 0)

    with Newtons_lock:
        Robot = RobotQueue.get()
        RobotQueue.task_done()
        RobotQueue.put(Robot)
    while checkData.wait(0) or not end.wait(0):
        try:
            scaleArray = Z1.get(True, 0.1)
            Z1.task_done()
            Weight     = scaleArray[1]
            avgWeight  = scaleArray[2]
        except Queue.Empty:
            pass
        GUIupdate.wait()
        try:
            touchDownPos     = touchDownQueue.get(True, 0.1)
            pastTouchDownPos = touchDownPos
            touchDownQueue.task_done()
        except Queue.Empty:
            touchDownPos = pastTouchDownPos

        UI       = np.zeros([height, width, 3])
        UI[:, :] = white

        if Robot.kg - Robot.error < Weight < Robot.kg + Robot.error:
            UI[height / 4:width / 2, :width / 2] = green       # Weight
        else:
            UI[height / 4:width / 2, :width / 2] = red         # Weight
        if Robot.kg - Robot.error < avgWeight < Robot.kg + Robot.error:
            UI[height / 4:width / 2, width / 2:] = green       # avgWeight
        else:
            UI[height / 4:width / 2, width / 2:] = red         # avgWeight
        if Robot.kg + 0.95 > 5500:
            UI[int(height / 1.333):, :width / 2] = red       # total weight
        else:
            UI[int(height / 1.333):, :width / 2] = green     # total weight

        UI[98:102, :]  = black
        UI[198:202, :] = black
        UI[298:302, :] = black
        UI[:, 198:202] = black

        upperCheck       = cv2.getTextSize(str(Robot.kg + Robot.error), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        cv2.putText(UI, str(Robot.kg + Robot.error), ((width / 4) - (upperCheck[0][0] / 2), (height / 8) + (upperCheck[0][1] / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        WeightCheck      = cv2.getTextSize("%.3f" % Weight, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        cv2.putText(UI, "%.3f" % Weight, ((width / 4) - (WeightCheck[0][0] / 2), ((height / 8) + (WeightCheck[0][1] / 2)) + height / 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        lowerCheck       = cv2.getTextSize(str(Robot.kg - Robot.error), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        cv2.putText(UI, str(Robot.kg - Robot.error), ((width / 4) - (lowerCheck[0][0] / 2), ((height / 8) + (lowerCheck[0][1] / 2)) + height / 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        totalWeightCheck = cv2.getTextSize("%.1f" % (Weight + 4060), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        cv2.putText(UI, "%.1f" % (Weight + 4060), ((width / 4) - (totalWeightCheck[0][0] / 2), ((height / 8) + (totalWeightCheck[0][1] / 2)) + int(height / 1.333)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

        sizeAvg          = cv2.getTextSize("Avg: %.1f" % avgWeight, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        cv2.putText(UI, "Avg: %.1f" % avgWeight, ((int(width / 1.333)) - (sizeAvg[0][0] / 2), (height / 8) + (sizeAvg[0][1] / 2) + height / 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        heightTextSize   = cv2.getTextSize("Z: %.2f" % touchDownPos[0][2], cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        cv2.putText(UI, "Z: %.2f" % touchDownPos[0][2], ((int(width / 1.333)) - (heightTextSize[0][0] / 2), (height / 8) + (heightTextSize[0][1] / 2) + int(height / 1.333)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

        GUIupdate.clear()

        cv2.imshow("UI", UI)
        cv2.moveWindow("UI", 50, 50)
        k = cv2.waitKey(10) & 0xFF
        if k == 115:
            saveImgEvent.clear()
        if k == 27:
            break
    with print_lock:
        end.set()
        print "No Data2"
    cv2.destroyAllWindows()


def writeList2File(textFile, DATA):
    with open(textFile, "a") as file:
        DATA = (','.join(map(str, DATA)))+'\n'
        file.write(DATA)


def main():
    connected.clear()
    saveImgEvent.set()
    degree = "*"
    unit = "mm"

    Robot = ABBProgram()
    RobotQueue.put(Robot)

    if StartEvent.wait(0):
        SocketConnectThread = threading.Thread(target=Scale, name="Scales")
        SocketConnectThread.daemon = True
        SocketConnectThread.start()

        connected.wait()
        checkData.wait(10)

        GetWeight = threading.Thread(target=Robot.getWeight, name="getWeight")
        GetWeight.daemon = True
        GetWeight.start()

        Z.wait()

        TakePicture = threading.Thread(target=takeImage, name="Take Picture")
        TakePicture.daemon = True
        TakePicture.start()

        GUIthread = threading.Thread(target=GUI, name="GUI")
        GUIthread.daemon = True
        GUIthread.start()

        [Robot.runMain(unit, i) for i in xrange(2)]
        [Robot.runMain(degree, i) for i in xrange(3, 6)]

        Robot.Home()
        Robot.printDateTime()
        print >> Robot.f, "Program Finished"
    with print_lock:
        raw_input('Close program?')
    Robot.close()
    time.sleep(2)
    # end.wait()
    print threading.enumerate()
    # Robot.f.close()

    # Robot.close()

thread = threading.Thread(target=main, name="Main")
thread.start()
