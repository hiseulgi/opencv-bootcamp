import cv2

image = cv2.imread("coca-cola-logo.png")
print(image.shape)

while True:
    cv2.imshow("Test", image)
    keypress = cv2.waitKey(1)
    if keypress == ord('q'):
        break

cv2.destroyAllWindows()
