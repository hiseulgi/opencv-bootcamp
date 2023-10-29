import cv2

image = cv2.imread("Apollo_11_Launch.jpg")

cv2.line(image, (200, 200), (600, 200), (255, 0, 0), 5, cv2.LINE_AA)

cv2.rectangle(image, (200, 300), (400, 500), (0, 255, 255), 5, cv2.LINE_AA)

cv2.circle(image, (300, 400), 100, (0, 0, 255), 5, cv2.LINE_AA)

cv2.putText(image, "viral!", (200, 550), cv2.FONT_HERSHEY_PLAIN,
            -1, (0, 0, 255), 2, cv2.LINE_AA)
while True:
    cv2.imshow("test", image)
    keypress = cv2.waitKey(1)
    if keypress == ord("q"):
        break

cv2.destroyAllWindows()
