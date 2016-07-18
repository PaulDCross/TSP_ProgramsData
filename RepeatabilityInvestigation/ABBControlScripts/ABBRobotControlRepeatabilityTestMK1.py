from datetime import datetime
import matplotlib.pyplot as plt
import transformations as t
import threading
import serial
import socket
import numpy
import Queue
import copy
import math
import time
import abb
import cv2
import sys
import os

saveImgEvent = threading.Event()
end          = threading.Event()
robotLock    = threading.Lock()
print_lock   = threading.Lock()


def y(x):
    return round((0.00000000000000001460497634638*x**6 - 0.0000000000000566584131607*x**5 + 0.0000000000899771179172776*x**4 - 0.000000084971113955370**3 + 0.0000527666092877035*x**2 - 0.025995092449057*x + 392.32063493137), 1)

def writeList2File(textFile, DATA):
    with open(textFile, "w") as file:
        DATA = '\n'.join(','.join(map(str, j)) for j in DATA)
        file.write(DATA)

def printDateTime():
    global f, startclock
    print >> f, "\n", datetime.now().strftime('%d-%m-%Y %H:%M:%S'), "\t%.5f" % (time.time() - startclock), "\t\t",

def makedir(DIR):
    if not os.path.exists(DIR):
        os.makedirs(DIR)
        time.sleep(0.5)

def goto(R, pos):
    while (R.get_cartesian() != pos):
        R.set_cartesian(pos)
        time.sleep(1)
        a = R.get_cartesian()
        print a, pos
    return R.get_cartesian()

def takePicture():
    global end, saveImgEvent
    cam = cv2.VideoCapture(0)   # Start the webcam
    cam.set(3, 1200)            # horizontal pixels
    cam.set(4, 720)             # vertical pixels
    while cam.isOpened() and not end.wait(0):
        _, frame = cam.read()
        if not saveImgEvent.wait(0):
            saveImage(frame)
    cam.release()
    cv2.destroyAllWindows()

def saveImage(frame):
    global f, DIR
    print "Saving Image"
    directory = os.path.join(DIR, "Images")
    makedir(directory)
    name = int(1 + len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))]))
    name = os.path.join(directory, "%003d" % name) + ".png"
    cv2.imwrite(name, frame)
    with print_lock:
        printDateTime()
        print >> f, "Picture saved:", name
        f.flush()
    print "Picture saved:", name
    saveImgEvent.set()

def r2d(radians):
    radians = [round((radians[i] * 180) / math.pi, 3) for i in range(3)]
    return radians

def Home(HomePos):
    global R, f
    with robotLock:
        R.set_speed(speed=[10, 10, 50, 50])
        with print_lock:
            print "Speeding up to 10%"
            printDateTime()
            print >> f, "Heading to Home Position",
            f.flush()
        while (R.get_joints() != HomePos):
            R.set_joints(HomePos)
            time.sleep(0.5)
            with print_lock:
                a = R.get_joints()
                print a, HomePos
        R.set_speed(speed=[1, 1, 50, 50])
        with print_lock:
            print "Slowing Down to 1%"
        cartesian    = R.get_cartesian()
        euler        = r2d(list(t.euler_from_quaternion(cartesian[1])))
        euler        = [round(euler[i], 1) for i in range(len(euler))]
        cartesian[1] = euler
        with print_lock:
            printDateTime()
            print >> f, "Arrived at Home Position: ", cartesian
            f.flush()

def Hover(HoverPosition):
    global R, f
    with robotLock:
        R.set_speed(speed=[10, 10, 50, 50])
        with print_lock:
            print "Speeding up to 10% 'hover'"
            printDateTime()
            print >> f, "Heading to Z coordinate %.1fmm" % HoverPosition[0][2],
            f.flush()
        while (R.get_cartesian() != HoverPosition):
            R.set_cartesian(HoverPosition)
            time.sleep(1)
            with print_lock:
                a = R.get_cartesian()
                print a, HoverPosition
        R.set_speed(speed=[1, 1, 50, 50])
        with print_lock:
            print "Slowing Down to 1% 'hover'"
        cartesian    = R.get_cartesian()
        euler        = r2d(list(t.euler_from_quaternion(cartesian[1])))
        euler        = [round(euler[i], 1) for i in range(len(euler))]
        cartesian[1] = euler
        with print_lock:
            printDateTime()
            print >> f, "Hovering at: ", cartesian,
            f.flush()
        time.sleep(0.2)
        saveImgEvent.clear()
        saveImgEvent.wait(1)

def main():
    global R, DIR, startclock, f
    end.clear()
    saveImgEvent.set()
    HomePos  = [0.89, 10.77, -26.32, -0.30, 85.55, 0.86]
    HoverPos = [[390, 0, 345], [0.174, 0, 0.985, 0]]
    Position = copy.deepcopy(HoverPos)

    while True:
        try:
            IP = raw_input('What is the IP of the Robot Controller: ')
            print "Connecting..."
            R = abb.Robot(ip=IP)
            print "Connected."
            break
        except (socket.error, socket.timeout):
            pass
        print "Incorrect IP, please try again.\n"

    while True:
        Start = raw_input('Start sequence? y/n: ')
        if Start in ("y", "n"):
            break
        else:
            print "Incorrect input, please try again.\n"

    if Start == "n":
        R.close()
        sys.exit(0)
        StartEvent.clear()

    DIR = os.path.join("TSP_Pictures", "RepeatabilityTest")
    makedir(DIR)
    numFolders = int(1 + len([name for name in os.listdir(DIR) if os.path.isdir(os.path.join(DIR, name))]))
    DIR = os.path.join(DIR, "%02d" % numFolders)
    makedir(DIR)
    # Create the log file and open it.
    with print_lock:
        LogFile  = os.path.join(DIR, "LogFile") + ".txt"
        f = open(LogFile, 'w')

    TakePicture = threading.Thread(target=takePicture, name="Take Picture")
    TakePicture.daemon = True
    TakePicture.start()

    startclock = time.time()
    with print_lock:
        printDateTime()
        print >> f, "Running program on Robot with IP address: ", IP

    Home(HomePos)

    while round(Position[0][2], 1) >= 335:
        Hover(HoverPos)
        Position[0][2] -= 0.1
        Position[0][2] = round(Position[0][2], 1)
        with print_lock:
            printDateTime()
            print >> f, "Target: ", list(([round(i,1) for i in Position[0]], Position[1])),
        result = goto(R, list(([round(i,1) for i in Position[0]], Position[1])))
        with print_lock:
            printDateTime()
            print >> f, "Result: ", result,
        time.sleep(0.2)
        saveImgEvent.clear()
        if not saveImgEvent.wait(1):
            break
        time.sleep(1)
        print "Finished Movement"

    Home(HomePos)

    with print_lock:
        printDateTime()
        print >> f, "Program Finished"
    R.close()
    end.set()

thread = threading.Thread(target=main, name="Main")
thread.start()
