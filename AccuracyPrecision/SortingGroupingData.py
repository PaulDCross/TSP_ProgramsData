import numpy as np
import sys, os
from operator import itemgetter
from itertools import groupby

DIR  = os.path.join("..", "..", "Python", "TSP_Testing", "TSP_Testing", "ProgramsData", "TSP_Pictures", "NewPillowRepeatabilityTest", "EE167.5", "358.0mm")
data = np.load(os.path.join(DIR, "datalineJuly14110.npy"))
