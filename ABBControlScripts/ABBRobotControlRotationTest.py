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
robotLock        = threading.Lock()
print_lock       = threading.Lock()
connected        = threading.Event()
checkData        = threading.Event()
StartEvent       = threading.Event()
saveImgEvent     = threading.Event()
ok               = threading.Event()
end              = threading.Event()
dic              = Queue.LifoQueue()
touchDownQueue   = Queue.LifoQueue()
RobotQueue       = Queue.LifoQueue()


class ABBProgram:
    """docstring for ABBProgram"""
    def __init__(self):
        degree                  = "*"
        unit                    = "mm"
        self.distance           = 10
        self.height, self.Step, IP, HH, Tool = self.initialize()
        self.homePos            = [0.00, 9.53, -22.78, 0.00, 83.25, 0.00]
        self.homePosCart        = [[400.0, 0.0, 450.0], [0.173956, 0.0, 0.984753, 0.0]]
        self.HoverPosition      = [380.0, 0.0, HH, 180.0, 20.0, 180.0]
        self.movType            = ["X", "Y", "Z", "Rx", "Ry", "Rz"]
        dictionary              = OrderedDict()
        dictionary["cartesian"] = OrderedDict()
        d                       = dictionary["cartesian"]
        self.error              = 5
        # Make dictionary.
        [self.get_coordinates(d, i, self.distance, self.Step, copy.deepcopy(self.HoverPosition)) for i in xrange(6)]
        dic.put(d)
        print([len(d[i]) for i in self.movType])
        print self.HoverPosition
        # Do you wish to start the experment of do you wish to abort now.
        while True:
            Start = raw_input('Start sequence? y/n: ')
            if Start in ("y", "n"):
                break
            else:
                print "Incorrect input, please try again.\n"
        # Convert self.HoverPosition from euler to quaternion.
        X                  = copy.deepcopy(self.HoverPosition)
        self.HoverPosition = self.d2q(X, 5)
        self.HoverPosition = [self.HoverPosition[:3], self.HoverPosition[3:]]

        if Start == "y":
            DIR = os.path.join("TSP_Pictures", "RotationTests", "RotationTest%.1f" % Tool[0][2], "%.1fmm" % self.height)
            self.makedir(DIR)
            self.numFolders = int(1 + len([name for name in os.listdir(DIR) if os.path.isdir(os.path.join(DIR, name))]))
            self.DIR        = os.path.join(DIR, "%02d" % self.numFolders)
            self.makedir(self.DIR)
            # Create the log file and open it.
            LogFile = os.path.join(self.DIR, "LogFile") + ".txt"
            self.f  = open(LogFile, 'w')
            StartEvent.set()
            self.startclock = time.time()
            self.printDateTime()
            print >> self.f, "Running program on Robot with IP address: ", IP,
            self.printDateTime()
            print >> self.f, "With tool: ", Tool
            self.Home()
        else:
            StartEvent.clear()

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
            HH = 390.0
        elif Tool[0][2] == 142.5:
            HH = 380.0
        elif Tool[0][2] == 155:
            HH = 370.0
        elif Tool[0][2] == 167.5:
            HH = 360.0
        elif Tool[0][2] == 180:
            HH = 340.0
        else:
            HH = 400.0
        print Tool
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

    def y(self, x):
        return round(((0.00000000000000001460497634638*x**6) - (0.0000000000000566584131607*x**5) + (0.0000000000899771179172776*x**4) - (0.000000084971113955370*x**3) + (0.0000527666092877035*x**2) - (0.025995092449057*x) + 392.32063493137 - 52), 1)

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

    def get_coordinates(self, d, i, distance, step, touchDownPos):
        XMovement = []
        touchDownPos[2] = self.height
        for newValue in self.linspace(touchDownPos[i], touchDownPos[i] + distance, step, "+"):
            # print newValue
            X    = copy.deepcopy(touchDownPos)
            X[i] = round(newValue, 1)
            XMovement.append(self.d2q(X, 5))
        for newValue in self.linspace(touchDownPos[i], touchDownPos[i] - distance, step, "-"):
            # print newValue
            X    = copy.deepcopy(touchDownPos)
            X[i] = round(newValue, 1)
            XMovement.append(self.d2q(X, 5))
        d[self.movType[i]] = XMovement
        return d

    def Home(self):
        with robotLock:
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
            euler        = self.r2d(t.euler_from_quaternion(cartesian[1]))
            euler        = [round(euler[i], 1) for i in range(len(euler))]
            cartesian[1] = euler
            self.printDateTime()
            print >> self.f, "Arrived at Home Position: ", cartesian
            self.f.flush()

    def hover(self):
        with robotLock:
            self.R.set_speed(speed=[10, 10, 50, 50])
            with print_lock:
                print "Speeding up to 10% 'hover'"
            self.printDateTime()
            print >> self.f, "Heading to Z coordinate %.1fmm" % self.HoverPosition[0][2],
            print "Heading to Z coordinate %.1fmm" % self.HoverPosition[0][2]
            self.f.flush()
            count = 0
            while (self.R.get_cartesian() != self.HoverPosition):
                self.R.set_cartesian(self.HoverPosition)
                time.sleep(1)
                if count >= 3:
                    self.R.set_speed(speed=[0.5, 0.5, 50, 50])
                with print_lock:
                    a = self.R.get_cartesian()
                    print a, self.HoverPosition
                count += 1
            self.R.set_speed(speed=[0.5, 0.5, 50, 50])
            with print_lock:
                print "Slowing Down to 1% 'hover'"
            cartesian    = self.R.get_cartesian()
            euler        = self.r2d(t.euler_from_quaternion(cartesian[1]))
            euler        = [round(euler[i], 1) for i in range(len(euler))]
            cartesian[1] = euler
            self.printDateTime()
            print >> self.f, "Hovering at: ", cartesian
            self.f.flush()
            time.sleep(0.2)
            saveImgEvent.clear()
            if not saveImgEvent.wait(1):
                print "Camera not working"

    def runMain(self, units, i):
        if not end.wait(0):
            with Newtons_lock:
                d = dic.get()
                dic.task_done()
                dic.put(d)
                data = d[self.movType[i]]
                DIR  = os.path.join(self.DIR, self.movType[i], "P")
                self.makedir(DIR)
                self.directory = Queue.LifoQueue()
                self.directory.put(DIR)
            # print >> self.f, "\n############################################################################################################"
            self.printDateTime()
            print >> self.f, "Starting Sequence: %s 0%s --> +20%s" % (DIR, units, units)
            self.hover()
            time.sleep(1)
            for dataline in data[:len(data) / 2]:
                dataline = [dataline[:3], dataline[3:]]
                with robotLock:
                    euler    = copy.deepcopy(dataline)
                    euler[1] = self.r2d(t.euler_from_quaternion(euler[1]))
                    euler[1] = [round(euler[1][i], 1) for i in range(len(euler[1]))]
                    self.printDateTime()
                    print >> self.f, "Target: ", euler,
                    print "Starting Movement"
                    while ([self.R.get_cartesian()[0], self.r2d(t.euler_from_quaternion(self.R.get_cartesian()[1]))] != euler) and not end.wait(0): # [dataline[0], [round(line,3) for line in dataline[1]]]
                        self.R.set_cartesian(dataline)
                        time.sleep(0.5)
                        with print_lock:
                            a = self.R.get_cartesian()
                            b = [self.R.get_cartesian()[0], self.r2d(t.euler_from_quaternion(self.R.get_cartesian()[1]))]
                            print a, dataline # [dataline[0], [round(line,3) for line in dataline[1]]]
                            print b, euler
                            print [[abs(c1) for c1 in b[0]], [abs(c2) for c2 in b[1]]] == [[abs(e1) for e1 in euler[0]], [abs(e2) for e2 in euler[1]]]
                        if [[abs(c1) for c1 in b[0]], [abs(c2) for c2 in b[1]]] == [[abs(e1) for e1 in euler[0]], [abs(e2) for e2 in euler[1]]]:
                            break
                    euler    = self.R.get_cartesian()
                    euler[1] = self.r2d(t.euler_from_quaternion(euler[1]))
                    euler[1] = [round(euler[1][i], 1) for i in range(len(euler[1]))]
                    self.printDateTime()
                    print >> self.f, "Result: ", euler,
                    self.f.flush()
                    with print_lock:
                        print "Result: ", euler
                    print "Finished Movement"
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
                dataline = [dataline[:3], dataline[3:]]
                with robotLock:
                    euler    = copy.deepcopy(dataline)
                    euler[1] = self.r2d(t.euler_from_quaternion(euler[1]))
                    euler[1] = [round(euler[1][i], 1) for i in range(len(euler[1]))]
                    self.printDateTime()
                    print >> self.f, "Target: ", euler,
                    print "Starting Movement"
                    while ([self.R.get_cartesian()[0], self.r2d(t.euler_from_quaternion(self.R.get_cartesian()[1]))] != euler) and not end.wait(0): # [dataline[0], [round(line,3) for line in dataline[1]]]
                        self.R.set_cartesian(dataline)
                        time.sleep(0.5)
                        with print_lock:
                            a = self.R.get_cartesian()
                            b = [self.R.get_cartesian()[0], self.r2d(t.euler_from_quaternion(self.R.get_cartesian()[1]))]
                            print a, dataline # [dataline[0], [round(line,3) for line in dataline[1]]]
                            print b, euler
                            print [[abs(c1) for c1 in b[0]], [abs(c2) for c2 in b[1]]] == [[abs(e1) for e1 in euler[0]], [abs(e2) for e2 in euler[1]]]
                        if [[abs(c1) for c1 in b[0]], [abs(c2) for c2 in b[1]]] == [[abs(e1) for e1 in euler[0]], [abs(e2) for e2 in euler[1]]]:
                            break
                    euler    = self.R.get_cartesian()
                    euler[1] = self.r2d(t.euler_from_quaternion(euler[1]))
                    euler[1] = [round(euler[1][i], 1) for i in range(len(euler[1]))]
                    self.printDateTime()
                    print >> self.f, "Result: ", euler,
                    self.f.flush()
                    with print_lock:
                        print "Result: ", euler
                    print "Finished Movement"
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


def saveImage(frame, image):
    with Newtons_lock:
        Robot = RobotQueue.get()
        DIR   = Robot.directory.get()
        RobotQueue.task_done()
        RobotQueue.put(Robot)
        Robot.directory.task_done()
        Robot.directory.put(DIR)
    dirc = os.path.join(DIR, "External")
    Robot.makedir(dirc)
    nameE = int(1 + len([nameE for nameE in os.listdir(dirc) if os.path.isfile(os.path.join(dirc, nameE))]))
    nameE = os.path.join(dirc, "%003d" % nameE) + ".png"
    cv2.imwrite(nameE, image)
    dirc = os.path.join(DIR, "Internal")
    Robot.makedir(dirc)
    name = int(1 + len([name for name in os.listdir(dirc) if os.path.isfile(os.path.join(dirc, name))]))
    name = os.path.join(dirc, "%003d" % name) + ".png"
    cv2.imwrite(name, frame)
    Robot.printDateTime()
    print >> Robot.f, "Picture saved:", name
    Robot.f.flush()
    with print_lock:
        print "Picture saved:", name
    saveImgEvent.set()


def takeImage():
    cam1 = cv2.VideoCapture(1)   # Start the webcam
    cam2 = cv2.VideoCapture(0)   # Start the webcam
    cam1.set(3, 1200)            # horizontal pixels
    cam1.set(4, 720)             # vertical pixels
    print "Cameras Opened"
    while cam1.isOpened() and cam2.isOpened() and not end.wait(0):
        _, frame = cam1.read()
        img      = cam2.read()[1]
        (h, w)   = img.shape[:2]
        M        = cv2.getRotationMatrix2D((w / 2, h / 2), -90, 1)
        image    = cv2.warpAffine(img, M, (w, h))[:, 80:w-80]
        if not saveImgEvent.wait(0):
            saveImage(frame, image)
    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()


def writeList2File(textFile, DATA):
    with open(textFile, "a") as file:
        DATA = (','.join(map(str, DATA)))+'\n'
        file.write(DATA)


def main():
    ok.clear()
    saveImgEvent.set()
    degree = "*"
    unit   = "mm"

    Robot  = ABBProgram()
    RobotQueue.put(Robot)

    if StartEvent.wait(0):
        TakePicture        = threading.Thread(target=takeImage, name="Take Picture")
        TakePicture.daemon = True
        TakePicture.start()

        # [Robot.runMain(unit, i) for i in xrange(1)]
        [Robot.runMain(degree, i) for i in xrange(3, 5)]

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
