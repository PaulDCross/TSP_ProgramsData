import numpy as np
import os
from itertools import groupby
import time
import scipy.io as sio

def getKey(item):
    return item[0]['Z']

DIR   = os.path.join("..", "..", "Python", "TSP_Testing", "TSP_Testing", "ProgramsData", "TSP_Pictures", "NewPillowRepeatabilityTest", "EE167.5", "358.0mm")
data  = np.load(os.path.join(DIR, "datalineJuly14110.npy"))
names = data.dtype.names
print names

sortedData  = np.array(sorted(data, key=getKey))                            # Sort data
groupedData = np.array([list(j) for i,j in groupby(sortedData, key=getKey)])# Group sorted data

# Calculating the Mean Absoloute Deviation  The average (absolute) distance from the mean
# group['NewXcoord'][:,i]                   gets all the values across the group for pin i
# len(group['NewXcoord'][0,:])              gets the number of pins in the image
# Every entry to the array is a different depth. Every value in every entry is the MAD for the pin at that depth.
print "Computing..."
StartTime = time.time()
madX = np.array([[np.mean(abs(group['NewXcoord'][:,i] - np.mean(group['NewXcoord'][:,i]))) for i in range(len(group['NewXcoord'][0,:]))] for group in groupedData])
madY = np.array([[np.mean(abs(group['NewYcoord'][:,i] - np.mean(group['NewYcoord'][:,i]))) for i in range(len(group['NewYcoord'][0,:]))] for group in groupedData])
madDispl = np.array([[np.mean(abs(group['Displacement'][:,i] - np.mean(group['Displacement'][:,i]))) for i in range(len(group['Displacement'][0,:]))] for group in groupedData])
print time.time() - StartTime
print "Finished."

print "Saving..."
StartTime = time.time()
sio.savemat("madXY.mat", {'madX' : madX, 'madY' : madY, "madDispl" : madDispl})
print time.time() - StartTime
print "Finished."
