import numpy as np
import numpy.ma as ma
import os
from itertools import groupby
import time
import scipy.io as sio
np.set_printoptions(precision=3, suppress=True, linewidth = 150)


DIR   = os.path.join("..", "..", "Python", "TSP_Testing", "TSP_Testing", "ProgramsData", "TSP_Pictures", "NewPillowRepeatabilityTest", "EE167.5", "358.0mm")
data  = np.load(os.path.join(DIR, "datalineJuly14110.npy"))
names = data.dtype.names
print names

sortedData  = sorted(np.ravel(data), key=lambda l: (l['Pin'], l['PresentImage']))   # Sort data by pin number then image number
groupedData = [list(j) for i,j in groupby(sortedData, key=lambda l: l[names.index('PresentImage')])] # group grouped data by image number
groupedgroupedData = [list(j) for i,j in groupby(groupedData, key=lambda l: l[0][names.index('Pin')])]      # Group sorted data by pin number

# [np.mean(abs([line[names.index('NewXcoord')] for line in paragraph] - np.mean([line[names.index('NewXcoord')] for line in paragraph]))) for paragraph in groupedgroupedData[0]]
"""
Calculating the Mean Absoloute Deviation  The average (absolute) distance from the mean
group['NewXcoord'][:,i]                   gets all the values across the group for pin i
len(group['NewXcoord'][0,:])              gets the number of pins in the image
Every entry to the array is a different depth. Every value in every entry is the MAD for the pin at that depth.
"""

# print "Computing..."
# StartTime = time.time()
# madX = np.array([[np.mean(  abs([line[names.index('NewXcoord')] for line in paragraph] - np.mean([line[names.index('NewXcoord')] for line in paragraph]))   ) for paragraph in essay] for essay in groupedgroupedData])
# madY = np.array([[np.mean(abs([line[names.index('NewYcoord')] for line in paragraph] - np.mean([line[names.index('NewYcoord')] for line in paragraph]))) for paragraph in essay] for essay in groupedgroupedData])
# madDispl = np.array([[np.mean(abs([line[names.index('Displacement')] for line in paragraph] - np.mean([line[names.index('Displacement')] for line in paragraph]))) for paragraph in essay] for essay in groupedgroupedData])
# print time.time() - StartTime
# print "Finished."

# print "Saving..."
# StartTime = time.time()
# sio.savemat("madXY.mat", {'madX' : madX, 'madY' : madY, "madDispl" : madDispl})
# print time.time() - StartTime
# print "Finished."



X = np.array([[ma.masked_array([pin[names.index('NewXcoord')] for pin in Data], [not pin[names.index('State')] for pin in Data]).compressed() for Data in GData] for GData in groupedgroupedData])
Y = np.array([[ma.masked_array([pin[names.index('NewYcoord')] for pin in Data], [not pin[names.index('State')] for pin in Data]).compressed() for Data in GData] for GData in groupedgroupedData])
Z1 = np.array([[ma.masked_array([abs(pin[names.index('PresentImage')]) for pin in Data], [not pin[names.index('State')] for pin in Data]).compressed() for Data in GData] for GData in groupedgroupedData])
Z2 = np.array([[ma.masked_array([pin[names.index('Pin')] for pin in Data], [not pin[names.index('State')] for pin in Data]).compressed() for Data in GData] for GData in groupedgroupedData])

MADX = np.array([[ np.mean([abs(line[i] - line[j]) for i in range(len(line)) for j in range(len(line))]) for line in section] for section in np.array([np.array([c for c in b if len(c)>3]) for b in X if len([c for c in b if len(c)>3])>3])])
MADY = np.array([[ np.mean([abs(line[i] - line[j]) for i in range(len(line)) for j in range(len(line))]) for line in section] for section in np.array([np.array([c for c in b if len(c)>3]) for b in Y if len([c for c in b if len(c)>3])>3])])
depth = np.array([[np.mean(line) for line in section] for section in np.array([np.array([c for c in b if len(c)>3]) for b in Z1 if len([c for c in b if len(c)>3])>3])])
pinName = np.array([[np.mean(line) for line in section] for section in np.array([np.array([c for c in b if len(c)>3]) for b in Z2 if len([c for c in b if len(c)>3])>3])])


zMADXdepth      = np.array([map(list, zip(MADX[i], depth[i], pinName[i])) for i in range(len(MADX))])
raveledX        = [pinDepth for pinRun in zMADXdepth for pinDepth in pinRun]
sortedMADXdepth = sorted(raveledX, key=lambda l: l[1])
groupingX       = [list(j) for i,j in groupby(sortedMADXdepth, key=lambda l: l[1])]
MADXPINS        = np.array([(np.mean([pin[0] for pin in DEPTH]), np.mean([pin[1] for pin in DEPTH])) for DEPTH in groupingX])

zMADYdepth      = np.array([map(list, zip(MADY[i], depth[i], pinName[i])) for i in range(len(MADY))])
raveledY        = [pinDepth for pinRun in zMADYdepth for pinDepth in pinRun]
sortedMADYdepth = sorted(raveledY, key=lambda l: l[1])
groupingY       = [list(j) for i,j in groupby(sortedMADYdepth, key=lambda l: l[1])]
MADYPINS        = np.array([(np.mean([pin[0] for pin in DEPTH]), np.mean([pin[1] for pin in DEPTH])) for DEPTH in groupingY])

sio.savemat("ZippedMADXYDepthName2.mat", {'MADX' : MADX, 'Depth' : depth, "PinName" : pinName, "MaddeningX" : MADXPINS, "MaddeningY" : MADYPINS})
# np.array([np.array([c for c in b if len(c)>3]) for b in X if len([c for c in b if len(c)>3])>3])
# np.array([np.array([c for c in b if len(c)>3]) for b in Y if len([c for c in b if len(c)>3])>3])
