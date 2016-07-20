import numpy as np
import sys, os
from operator import itemgetter
from itertools import groupby

def getKey(item):
    return item[0]['Z']

DIR   = os.path.join("..", "..", "Python", "TSP_Testing", "TSP_Testing", "ProgramsData", "TSP_Pictures", "NewPillowRepeatabilityTest", "EE167.5", "358.0mm")
data  = np.load(os.path.join(DIR, "datalineJuly14110.npy"))
names = data.dtype.names

sortedData  = np.array(sorted(data, key=getKey))                            # Sort data
groupedData = np.array([list(j) for i,j in groupby(sortedData, key=getKey)])# Group sorted data


