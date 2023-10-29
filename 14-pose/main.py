import cv2
from cvzone.PoseModule import PoseDetector

source = cv2.VideoCapture(0)
detector = PoseDetector()

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

    # find pose and draw point + skeleton
    frame = detector.findPose(frame)

    # information extraction from above
    lm_list, bbox_info = detector.findPosition(frame)
    if bbox_info:
        print(lm_list)
        print(bbox_info)
        print("====")
        center = bbox_info["center"]
        cv2.circle(frame, center, 5, (255, 0, 0), -1)

    # * output final frame
    cv2.imshow(win_name, frame)

source.release()
cv2.destroyAllWindows()
