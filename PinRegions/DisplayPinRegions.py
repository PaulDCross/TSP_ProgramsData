import cv2
import os
from PillowEdited import *
import numpy as np

def detectorParameters():
    """Set up the blob detector parameters"""
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    params.minThreshold = 150;
    params.maxThreshold = 255;
    # Filter by Colour
    params.filterByColor = True
    params.blobColor = 255
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 90 # 120, 142
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.7 # 0.8
    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 4:
        detector = cv2.SimpleBlobDetector_create(params)
    else :
        detector = cv2.SimpleBlobDetector_create(params)
    return detector


refPt              = [(124, 83), (1057, 585)]
x1,y1              = refPt[0][0], refPt[0][1]
x2,y2              = refPt[1][0], refPt[1][1]
pathname1          = os.path.join("TSP_Pictures", "NewPillowRotationTest", "RotationTest167.5", "350.0mm", "01", "Rx", "P", "Internal", "%003d" % 1) + ".png"
img                = cv2.imread(pathname1)
init               = Pillow(img, refPt)
img                = img[y1:y2, x1:x2]
img                = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blur             = cv2.GaussianBlur(img, (5,5), 0)
# ret, thresholded = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
thresholded1       = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)[1]
image              = cv2.morphologyEx(thresholded1, cv2.MORPH_OPEN, None, iterations = 1)
ROI                = cv2.dilate(image, None, iterations = 4)
# Crop to ROI
key                = init.detectorParameters().detect(ROI)
# ROIkey             = cv2.drawKeypoints(ROI, key, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
ROIkey             = cv2.cvtColor(ROI, cv2.COLOR_GRAY2BGR)
Columns, Rows, crosspoints    = Pins(refPt).main(key)

for i in xrange(len(key)):
    # Label the Blobs
    size = cv2.getTextSize("%d" % (i+1), cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.putText(ROIkey, "%d" % (i+1), (int(crosspoints[i][0]-size[0][0]), int(crosspoints[i][1]-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)

[cv2.line(ROIkey, (0, crosspoints[i][1]), (crosspoints[i][0], crosspoints[i][1]), (150,150,150), 2) for i in xrange(14, len(crosspoints), 15)]
[cv2.line(ROIkey, (crosspoints[i][0], 0), (crosspoints[i][0], y2), (150,150,150), 2) for i in xrange(0, 14)]

print Columns, Rows
# cv2.imwrite("ROI1.png", ROI)
# cv2.imwrite("PinRegions.png", ROIkey)
cv2.imshow("Camera", ROIkey)
cv2.imshow("Camera1", ROI)
if cv2.waitKey(0) & 0xFF == 27:
    cv2.destroyAllWindows()
