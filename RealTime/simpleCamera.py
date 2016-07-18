import cv2
import numpy as np
import threading
import time
import sys

# cam1 = cv2.VideoCapture(0)
cam2 = cv2.VideoCapture(1)
cam2.set(3, 1200)            # horizontal pixels
cam2.set(4, 720)             # vertical pixels
while True:
    _, frame = cam2.read()
    # img = cam1.read()[1]
    # (h, w) = img.shape[:2]
    # M = cv2.getRotationMatrix2D((w / 2, h / 2), -90, 1)
    # image = cv2.warpAffine(img, M, (w, h))[:, 80:w-80]
    # cv2.imshow("Window1", image)
    cv2.imshow("Window2", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break
        # cam1.release()
        cam2.release()
        cv2.destroyAllWindows()

# def internal():
#     cam1 = cv2.VideoCapture(1)   # Start the webcam
#     # cam1.set(3, 320)             # horizontal pixels
#     # cam1.set(4, 240)             # vertical pixels
#     while cam1.isOpened():
#         _, frame = cam1.read()
#         cv2.imshow("Window1", frame)
#         if cv2.waitKey(5) & 0xFF == 27:
#             break
#             cv2.destroyAllWindows()
#     cam1.release()
#     cv2.destroyAllWindows()

# def external():
#     cam2 = cv2.VideoCapture(0)   # Start the webcam
#     # cam2.set(3, 320)             # horizontal pixels
#     # cam2.set(4, 240)             # vertical pixels
#     while cam2.isOpened():
#         img      = cam2.read()[1]
#         (h, w)   = img.shape[:2]
#         M        = cv2.getRotationMatrix2D((w / 2, h / 2), -90, 1)
#         image    = cv2.warpAffine(img, M, (w, h))[:, 80:w-80]
#         cv2.imshow("Window3", image)
#         if cv2.waitKey(5) & 0xFF == 27:
#             break
#             cv2.destroyAllWindows()
#     cam2.release()
#     cv2.destroyAllWindows()

# def main():
#     Internal             = threading.Thread(target=internal, name="internal")
#     Internal.daemon      = True
#     Internal.start()

#     External             = threading.Thread(target=external, name="external")
#     External.daemon      = True
#     External.start()

#     time.sleep(10)

#     print threading.enumerate()
#     sys.exit()

# thread = threading.Thread(target=main, name="Main")
# thread.start()

