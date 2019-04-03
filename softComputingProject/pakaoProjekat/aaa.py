import cv2
import numpy as np
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
pylab.rcParams['figure.figsize'] = 16, 12

snimak = cv2.VideoCapture('snimci/video7.mp4')

while (snimak.isOpened()):
    # uzmi frejm
    ret, frame = snimak.read()
    if ret == True:

        #img_crop = frame[150:450, 200:398]

        # transformacija iz bgr u rgb posto openCV ucitava kao bgr
        frejm = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frejm_grey = cv2.cvtColor(frejm, cv2.COLOR_RGB2GRAY)

        # retww je izracunata vrednost praga, image_bin je binarna slika
        retww, frejm_bin = cv2.threshold(frejm_grey, 48, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((3, 3))  # strukturni element 3x3 blok
        frejm_er = cv2.erode(frejm_bin, kernel, iterations=3)

        im, contours, hierarchy = cv2.findContours(frejm_er, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # print("Number of objects found = ", len(contours)-1)
        cv2.drawContours(im, contours, -1, (0, 255, 0), 3)

        cv2.imshow('Frame', im)
        plt.show()

        ljudi = []
        for contour in contours:
            ljudi.append(contour)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

            # Break the loop
    else:
        break

snimak.release()

# Closes all the frames
cv2.destroyAllWindows()