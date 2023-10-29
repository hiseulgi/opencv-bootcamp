import cv2
import sys

# get arg for camera id
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

# get video source from webcam (camera 0)
source = cv2.VideoCapture(s)

# new window
win_name = "Camera Preview"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# looping until q pressed
while cv2.waitKey(1) != ord("q"):
    # read frame on webcam
    has_frame, frame = source.read()

    # if camera didnt catch any frame, then exit
    if not has_frame:
        break

    # flip frame (mirror)
    frame_flipped = cv2.flip(frame, 1)

    # convert img to grayscale
    frame_gry = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2GRAY)

    # threshold global
    # retval, frame_thresh = cv2.threshold(frame_gry, 100, 255, cv2.THRESH_BINARY)

    # adaptive threshold
    frame_thresh_adp = cv2.adaptiveThreshold(
        frame_gry, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 7)

    # show frame by frame on windows
    cv2.imshow(win_name, frame_thresh_adp)

# ! release webcam in program (IMPORTANT)
source.release()

# exit windows
cv2.destroyWindow(win_name)
