import cv2
import numpy as np

capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    _, img = capture.read()
    img  = cv2.resize(img, (int(img.shape[1]*0.5), int(img.shape[0]*0.5)))

    img1= img
    img2 = cv2.flip(img, 0)
    img3 = cv2.flip(img, 1)
    img4 = cv2.flip(img, -1)
    


    Hori = np.concatenate((img3, img1), axis=1) 
    Hori2 = np.concatenate((img4, img2), axis=1)
    Verti = np.concatenate((Hori, Hori2), axis=0)

    cv2.imshow("frame", Verti)

    if cv2.waitKey(20) & 0xff == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()


