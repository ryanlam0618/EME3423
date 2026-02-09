import cv2 



capture = cv2.VideoCapture("Resources/dog.mp4")


while True:
    success, img = capture.read()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow("img", img)

    if cv2.waitKey(20) & 0xff == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()


"""print(cv2.__version__)

img = cv2.imread('Resources/BlackPink.png')

img = cv2.resize(img, (int(img.shape[1]/1.5), int(img.shape[0]/1.5)))



print(img.shape)
cv2.imshow('Output', img)

cv2.waitKey(0)"""