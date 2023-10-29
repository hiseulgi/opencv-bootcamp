import cv2
import numpy as np
import os

# import class labels
classFile = "coco_class_labels.txt"
with open(classFile) as fp:
    labels = fp.read().split("\n")

# read tensorflow model
modelFile = os.path.join(
    "models", "ssd_mobilenet_v2_coco_2018_03_29", "frozen_inference_graph.pb")
configFile = os.path.join("models", "ssd_mobilenet_v2_coco_2018_03_29.pbtxt")
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

# video source setting
source = cv2.VideoCapture(0)

win_name = "Object Detection Cam"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# const
FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1
INPUT_SIZE = (300, 300)
MEAN = (0, 0, 0)
CONFIDENCE_THRESHOLD = 0.5

# function utils


def display_text(im, text, x, y):
    # Get text size
    textSize = cv2.getTextSize(text, FONTFACE, FONT_SCALE, THICKNESS)
    dim = textSize[0]
    baseline = textSize[1]

    # Use text size to create a black rectangle
    cv2.rectangle(
        im,
        (x, y - dim[1] - baseline),
        (x + dim[0], y + baseline),
        (0, 0, 0),
        cv2.FILLED,
    )

    # Display text inside the rectangle
    cv2.putText(
        im,
        text,
        (x, y - 5),
        FONTFACE,
        FONT_SCALE,
        (0, 255, 255),
        THICKNESS,
        cv2.LINE_AA,
    )


# * loop on reading video source
while cv2.waitKey(1) != ord("q"):
    has_frame, frame = source.read()
    if not has_frame:
        break
    frame = cv2.flip(frame, 1)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # * detect object
    blob = cv2.dnn.blobFromImage(frame, 1.0, INPUT_SIZE, MEAN, True, False)
    net.setInput(blob)
    objects = net.forward()

    # * display detected object
    for i in range(objects.shape[2]):
        # find class and score
        classId = int(objects[0, 0, i, 1])
        confidence = float(objects[0, 0, i, 2])

        # get original coordinates from normalized coordinates
        x = int(objects[0, 0, i, 3] * frame_width)
        y = int(objects[0, 0, i, 4] * frame_height)
        w = int(objects[0, 0, i, 5] * frame_width - x)
        h = int(objects[0, 0, i, 6] * frame_height - y)

        # check score threshold
        if confidence > CONFIDENCE_THRESHOLD:
            label = "{}: %.2f".format(labels[classId]) % confidence
            display_text(frame, label, x, y)

            # display bounding box on object
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

    # * fps information
    t, _ = net.getPerfProfile()
    label = "Inference time: %.2f ms" % (t*1000.0/cv2.getTickFrequency())
    cv2.putText(frame, label, (0, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    # * show frame
    cv2.imshow(win_name, frame)

# * release all after done
source.release()
cv2.destroyAllWindows()
