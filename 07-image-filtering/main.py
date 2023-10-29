import cv2
import sys
import numpy

# const menu
PREVIEW = 0  # Preview Mode
BLUR = 1  # Blurring Filter
FEATURES = 2  # Corner Feature Detector
CANNY = 3  # Canny Edge Detector

# feature params for corner feature detector
feature_params = dict(
    maxCorners=500,
    qualityLevel=0.2,
    minDistance=15,
    blockSize=9
)

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

image_filter = PREVIEW
alive = True

win_name = "Camera Filters"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
result = None

source = cv2.VideoCapture(s)

while alive:
    has_frame, frame = source.read()

    frame = cv2.flip(frame, 1)

    # frame filter operation
    if image_filter == PREVIEW:
        result = frame
    elif image_filter == CANNY:
        result = cv2.Canny(frame, 64, 127)
    elif image_filter == BLUR:
        result = cv2.blur(frame, (5, 5))
    elif image_filter == FEATURES:
        result = frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(frame_gray, **feature_params)

        if corners is not None:
            for x, y in numpy.int0(corners).reshape(-1, 2):
                cv2.circle(result, (x, y), 10, (0, 255, 0), 1)

    # show final result
    cv2.imshow(win_name, result)

    # keypress handler
    key = cv2.waitKey(1)
    if key == ord("q"):
        alive = False
    elif key == ord("1"):
        image_filter = CANNY
    elif key == ord("2"):
        image_filter = BLUR
    elif key == ord("3"):
        image_filter = FEATURES
    elif key == ord("4"):
        image_filter = PREVIEW

source.release()
cv2.destroyAllWindows()
