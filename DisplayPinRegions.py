from libraries.MachineVisionAndmore.PillowEdited import *
import numpy as np
import cv2
import os
import copy

refPt                      = [(124, 83), (1057, 585)]
x1,y1                      = refPt[0][0], refPt[0][1]
x2,y2                      = refPt[1][0], refPt[1][1]
ProgramsData               = os.path.join("..", "Python", "TSP_Testing", "TSP_Testing", "ProgramsData")
pathname1                  = os.path.join(ProgramsData, "TSP_Pictures", "NewPillowRotationTest", "RotationTest167.5", "350.0mm", "01", "Rx", "P", "Internal", "%003d" % 1) + ".png"
# Import Image
img                        = cv2.imread(pathname1)
# Initialise Pillow Class
init                       = Pillow(img, refPt)
# Get the region of interest of the image
ROI, _                     = init.getFrame()
# Initialise the Pin regions
Columns, Rows, crosspoints = Pins(refPt).main(init.detectorParameters().detect(ROI))
# Read in the pin regions
xyn                        = rw().readFile2List("Pin_Regions.txt")
data1                      = init.initialiseData(xyn)
# Get the Keypoints
key                        = init.detectorParameters().detect(ROI)

ROI        = cv2.cvtColor(ROI, cv2.COLOR_GRAY2BGR)
ROIregions = copy.deepcopy(ROI)
blackBar   = np.zeros((10, ROI.shape[1], 3), np.uint8)
ROI        = np.concatenate((ROI, blackBar), axis=0)

for i in xrange(len(key)):
    # Label the Blobs
    size = cv2.getTextSize("%d" % (i+1), cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.putText(ROIregions, "%d" % (i+1), (int(crosspoints[i][0]-size[0][0]), int(crosspoints[i][1]-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)

[cv2.line(ROIregions, (0, crosspoints[i][1]), (crosspoints[i][0], crosspoints[i][1]), (150,150,150), 2) for i in xrange(14, len(crosspoints), 15)]
[cv2.line(ROIregions, (crosspoints[i][0], 0), (crosspoints[i][0], y2), (150,150,150), 2) for i in xrange(0, 14)]


for pin in data1:
    # Label the Blobs
    size = cv2.getTextSize(str(int(pin[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.putText(ROI, str(int(pin[0])), (int(pin[1] - (size[0][0]/4)), int(pin[2] + 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    # cv2.circle(ROI, (int(pin[1]), int(pin[2])), 2, (0,0,255), 2)
    cv2.circle(ROI, (int(pin[1]), int(pin[2])), 10, (0,0,255), 3)

# print Columns, Rows
# cv2.imwrite("LabeledPins.png", ROI)
# cv2.imwrite("PinRegions.png", ROIregions)
cv2.imshow("Camera", ROIregions)
# cv2.imshow("Camera1", ROI)
if cv2.waitKey(0) & 0xFF == 27:
    cv2.destroyAllWindows()
