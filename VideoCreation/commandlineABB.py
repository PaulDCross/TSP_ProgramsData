import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../libraries/MachineVisionAndmore")
from PillowEdited import *
import abb

R = abb.Robot(ip='127.0.0.1')

speedVariable = 2
R.set_speed(speed=[speedVariable, speedVariable, 50, 50])
print R.get_cartesian()

print "R.set_cartesian([[380.0, 0.0, 450.0], [0.173956, 0.0, 0.984753, 0.0]])"
