from Pillow import *
import os
import numpy
from matplotlib import pyplot as plt


array0, array1, array2, array3, array4, array5, array6, array7, array8, array9, array10, array11, array12, cart, x, y, lists = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
rw = rw()
for i in range(101, 1, -1):
    datafile = rw.readFile2List(os.path.join("TSP_Pictures", "DataFiles", "338.2mm", "02", "X", "P", "Data_%d.txt" % i))
    array0.append(numpy.mean([data[0] for data in datafile]))
    array1.append(numpy.mean([data[1] for data in datafile]))
    array2.append(numpy.mean([data[2] for data in datafile]))
    array3.append(numpy.mean([data[3] for data in datafile]))
    array4.append(numpy.mean([data[4] for data in datafile]))
    array5.append(numpy.mean([data[5] for data in datafile]))
    array6.append(numpy.mean([data[6] for data in datafile]))
    array7.append(numpy.mean([data[7] for data in datafile]))
    array8.append(numpy.mean([data[8] for data in datafile]))
    array9.append(numpy.mean([data[9] for data in datafile]))
    array10.append(numpy.mean([data[10] for data in datafile]))
    array11.append(numpy.mean([data[11] for data in datafile]))
    array12.append(numpy.mean([data[12] for data in datafile]))
for i in range(1, 102):
    datafile = rw.readFile2List(os.path.join("TSP_Pictures", "DataFiles", "338.2mm", "02", "X", "N", "Data_%d.txt" % i))
    array0.append(numpy.mean([data[0] for data in datafile]))
    array1.append(numpy.mean([data[1] for data in datafile]))
    array2.append(numpy.mean([data[2] for data in datafile]))
    array3.append(numpy.mean([data[3] for data in datafile]))
    array4.append(numpy.mean([data[4] for data in datafile]))
    array5.append(numpy.mean([data[5] for data in datafile]))
    array6.append(numpy.mean([data[6] for data in datafile]))
    array7.append(numpy.mean([data[7] for data in datafile]))
    array8.append(numpy.mean([data[8] for data in datafile]))
    array9.append(numpy.mean([data[9] for data in datafile]))
    array10.append(numpy.mean([data[10] for data in datafile]))
    array11.append(numpy.mean([data[11] for data in datafile]))
    array12.append(numpy.mean([data[12] for data in datafile]))

def readFile2List(textFile, phrase):
    with open(textFile, "r") as file:
        data = []
        for line in file.readlines():
            if phrase in line:
                data.append([str(i) for i in line.split()])
    return data

run = readFile2List(os.path.join("TSP_Pictures", "SetHeightPictureCollection", "338.2mm", "02", "20160308_15_28_31.txt"), "Result")

for i in range((len(run)/2)-1, -1,-1):
    print i
    string = run[i][4]+run[i][5]+run[i][6]+run[i][7]+run[i][8]+run[i][9]
    print string
    strings = string.replace('[','').split('],')
    # print string
    lists.append([map(float, s.replace(']','').split(',')) for s in strings])
    # print lists
for i in range((len(lists)/2), (len(lists))):
    cart.append(lists[i][0][0])
for i in range(len(lists)/2-1, -1, -1):
    cart.append(lists[i][0][0])
for i in range(0,200,5):
    x.append([i, i, i, i, i, i, i])
    y.append([0.5, 1, 1.5, 2, 2.5, 3, 3.5])

fig = plt.figure()
plt.subplot(1,1,1)
plt.ylim([0, 5])
plt.xlim([0, 200])
plt.plot(array10)
plt.plot([val for sublist in x for val in sublist], [val for sublist in y for val in sublist])

fig = plt.figure()
plt.subplot(4,2,1)
# plt.ylim([0, 1])
# plt.xlim([0, 200])
plt.plot(array7)
plt.subplot(4,2,3)
# plt.ylim([0, 5])
# plt.xlim([0, 200])
plt.plot(array10)
plt.subplot(4,2,5)
# plt.ylim([0, 200])
# plt.xlim([0, 200])
plt.plot(array11)

array0, array1, array2, array3, array4, array5, array6, array7, array8, array9, array10, array11, array12 = [], [], [], [], [], [], [], [], [], [], [], [], []
for i in range(101,1,-1):
    datafile = rw.readFile2List(os.path.join("TSP_Pictures", "DataFiles", "339.0mm", "01", "X", "P", "Data_%d.txt" % i))
    array0.append(numpy.mean([data[0] for data in datafile]))
    array1.append(numpy.mean([data[1] for data in datafile]))
    array2.append(numpy.mean([data[2] for data in datafile]))
    array3.append(numpy.mean([data[3] for data in datafile]))
    array4.append(numpy.mean([data[4] for data in datafile]))
    array5.append(numpy.mean([data[5] for data in datafile]))
    array6.append(numpy.mean([data[6] for data in datafile]))
    array7.append(numpy.mean([data[7] for data in datafile]))
    array8.append(numpy.mean([data[8] for data in datafile]))
    array9.append(numpy.mean([data[9] for data in datafile]))
    array10.append(numpy.mean([data[10] for data in datafile]))
    array11.append(numpy.mean([data[11] for data in datafile]))
    array12.append(numpy.mean([data[12] for data in datafile]))
for i in range(1,102):
    datafile = rw.readFile2List(os.path.join("TSP_Pictures", "DataFiles", "339.0mm", "01", "X", "N", "Data_%d.txt" % i))
    array0.append(numpy.mean([data[0] for data in datafile]))
    array1.append(numpy.mean([data[1] for data in datafile]))
    array2.append(numpy.mean([data[2] for data in datafile]))
    array3.append(numpy.mean([data[3] for data in datafile]))
    array4.append(numpy.mean([data[4] for data in datafile]))
    array5.append(numpy.mean([data[5] for data in datafile]))
    array6.append(numpy.mean([data[6] for data in datafile]))
    array7.append(numpy.mean([data[7] for data in datafile]))
    array8.append(numpy.mean([data[8] for data in datafile]))
    array9.append(numpy.mean([data[9] for data in datafile]))
    array10.append(numpy.mean([data[10] for data in datafile]))
    array11.append(numpy.mean([data[11] for data in datafile]))
    array12.append(numpy.mean([data[12] for data in datafile]))

plt.subplot(4,2,2)
plt.ylim([0, 1])
plt.xlim([0, 200])
plt.plot(array7)
plt.subplot(4,2,4)
plt.ylim([0, 5])
plt.xlim([0, 200])
plt.plot(array10)
plt.subplot(4,2,6)
plt.ylim([0, 200])
plt.xlim([0, 200])
plt.plot(array11)
plt.subplot(4,2,7)
plt.xlim([0, 200])
plt.plot(cart)
plt.subplot(4,2,8)
plt.xlim([0, 200])
plt.plot(cart)
plt.subplot(4,2,3)
plt.xlim([0, 200])
plt.plot([val for sublist in x for val in sublist], [val for sublist in y for val in sublist])

plt.show()