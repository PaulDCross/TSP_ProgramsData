import numpy as np
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
groupedData = [list(j) for i,j in groupby(sortedData, key=lambda l: l[15])] # group grouped data by image number
groupedgroupedData = [list(j) for i,j in groupby(groupedData, key=lambda l: l[0][0])]      # Group sorted data by pin number

[np.mean(abs([line[names.index('NewXcoord')] for line in paragraph] - np.mean([line[names.index('NewXcoord')] for line in paragraph]))) for paragraph in groupedgroupedData[0]]

# Calculating the Mean Absoloute Deviation  The average (absolute) distance from the mean
# group['NewXcoord'][:,i]                   gets all the values across the group for pin i
# len(group['NewXcoord'][0,:])              gets the number of pins in the image
# Every entry to the array is a different depth. Every value in every entry is the MAD for the pin at that depth.
print "Computing..."
StartTime = time.time()
madX = np.array([[np.mean(abs([line[names.index('NewXcoord')] for line in paragraph] - np.mean([line[names.index('NewXcoord')] for line in paragraph]))) for paragraph in essay] for essay in groupedgroupedData])
madY = np.array([[np.mean(abs([line[names.index('NewYcoord')] for line in paragraph] - np.mean([line[names.index('NewYcoord')] for line in paragraph]))) for paragraph in essay] for essay in groupedgroupedData])
madDispl = np.array([[np.mean(abs([line[names.index('Displacement')] for line in paragraph] - np.mean([line[names.index('Displacement')] for line in paragraph]))) for paragraph in essay] for essay in groupedgroupedData])
print time.time() - StartTime
print "Finished."

print "Saving..."
StartTime = time.time()
sio.savemat("madXY.mat", {'madX' : madX, 'madY' : madY, "madDispl" : madDispl})
print time.time() - StartTime
print "Finished."
