import os
import cv2
import sys
from zipfile import ZipFile
from urllib.request import urlretrieve


# Download model face detection
# ========================-Downloading Assets-========================
def download_and_unzip(url, save_path):
    print(f"Downloading and extracting assests....", end="")

    # Downloading zip file using urllib package.
    urlretrieve(url, save_path)

    try:
        # Extracting zip file using the zipfile package.
        with ZipFile(save_path) as z:
            # Extract ZIP file contents in the same directory.
            z.extractall(os.path.split(save_path)[0])

        print("Done")

    except Exception as e:
        print("\nInvalid file.", e)


URL = r"https://www.dropbox.com/s/efitgt363ada95a/opencv_bootcamp_assets_12.zip?dl=1"

asset_zip_path = os.path.join(os.getcwd(), f"opencv_bootcamp_assets_12.zip")

# Download if assest ZIP does not exists.
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)
# ====================================================================

# argument parser
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

# video capture source
source = cv2.VideoCapture(s)

# new windows
win_name = "Camera Preview"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# load dnn model
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

# model parameters
in_width = 300
in_height = 300
mean = [104, 117, 123]
conf_threshold = 0.7

# main loop
while cv2.waitKey(1) != ord("q"):
    # read frame from video
    has_frame, frame = source.read()
    if not has_frame:
        break

    # flip frame and information parse
    frame = cv2.flip(frame, 1)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # create a 4d blob from a frame
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (in_width, in_height), mean, swapRB=False, crop=False)

    # run a model
    net.setInput(blob)
    detections = net.forward()

    # handle detected face
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # get face with good confidence only
        if confidence > conf_threshold:
            x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
            y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
            x_right_top = int(detections[0, 0, i, 5] * frame_width)
            y_right_top = int(detections[0, 0, i, 6] * frame_height)

            # bounding box on face
            cv2.rectangle(
                frame,
                (x_left_bottom, y_left_bottom),
                (x_right_top, y_right_top),
                (0, 255, 0)
            )

            # draw confidence label on face bounding box
            label = "Confidence: %.4f" % confidence
            label_size, base_line = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(
                frame,
                (x_left_bottom, y_left_bottom - label_size[1]),
                (x_left_bottom + label_size[0], y_left_bottom + base_line),
                (255, 255, 255),
                cv2.FILLED,
            )
            cv2.putText(
                frame,
                label,
                (x_left_bottom, y_left_bottom),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0)
            )

    # fps information
    t, _ = net.getPerfProfile()
    label = "Inference time: %.2f ms" % (t*1000.0/cv2.getTickFrequency())
    cv2.putText(frame, label, (0, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    # show frame
    cv2.imshow(win_name, frame)

# destroy all video and windows
source.release()
cv2.destroyAllWindows()
