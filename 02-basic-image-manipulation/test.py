import cv2

image = cv2.imread("New_Zealand_Boat.jpg")
resized_image = cv2.resize(image, None, fx=2, fy=2)

while True:
    cv2.imshow("Test", resized_image)
    keypress = cv2.waitKey(1)
    if keypress == ord("q"):
        break

cv2.destroyAllWindows()
