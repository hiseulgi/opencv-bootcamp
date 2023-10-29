import cv2
import os

# ! Program berjalan sangat lambat, karena komputasi pada predict sendi (point) sangat berat
# perlu waktu +-34s di notebook, apalagi disini yang realtime
# ---------------------------------------------------------------------------- #
# * MODEL SETUP
proto_file = "pose_deploy_linevec_faster_4_stages.prototxt"
weights_file = os.path.join("model", "pose_iter_160000.caffemodel")
n_points = 15
POSE_PAIRS = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [1, 5],
    [5, 6],
    [6, 7],
    [1, 14],
    [14, 8],
    [8, 9],
    [9, 10],
    [14, 11],
    [11, 12],
    [12, 13],
]

net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
net_input_size = (368, 368)

# ---------------------------------------------------------------------------- #

source = cv2.VideoCapture(0)

win_name = "Pose Detection"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

while cv2.waitKey(1) != ord("q"):
    # * read frame from video
    has_frame, frame = source.read()
    if not has_frame:
        break
    frame = cv2.flip(frame, 1)
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    # * model predict
    blob = cv2.dnn.blobFromImage(
        frame, 1.0 / 255, net_input_size, (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output = net.forward()

    # * extract points
    # find x and y scale
    scale_x = frame_width / output.shape[3]
    scale_y = frame_height / output.shape[2]

    # empty list to store detected keypoints
    points = []

    # threshold
    threshold = 0.1

    for i in range(n_points):
        # get probability map
        prob_map = output[0, i, :, :]

        # find global maxima of the probmap
        min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

        # scale point to fit on original image
        x = scale_x * point[0]
        y = scale_y * point[1]

        if prob > threshold:
            points.append((int(x), int(y)))
        else:
            points.append(None)

    # * draw skeleton
    for pair in POSE_PAIRS:
        part_a = pair[0]
        part_b = pair[1]

        if points[part_a] and points[part_b]:
            cv2.line(frame, points[part_a], points[part_b], (255, 255, 0), 2)
            cv2.circle(frame, points[part_a], 8, (255, 0, 0), -1, cv2.FILLED)

    # * output final frame
    cv2.imshow(win_name, frame)

source.release()
cv2.destroyAllWindows()
