import cv2
import numpy as np
import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 16, 12

file = open("out.txt", "w")
file.write("RA168/2015,Marko Katic"+'\n')
file.write("file,count"+'\n')

for video in range(1,11):
    #kreira VideoCapture objekat i cita iz datog fajla
    snimak = cv2.VideoCapture('snimci/video' + str(video) + '.mp4')
    ukupno = 0
    i = 0

    # Citaj dok ima frejmova
    while (snimak.isOpened()):

        # uzmi frejm
        ret, frame = snimak.read()

        i = i + 1
        if i % 35 == 0:

            #provera da li frejm postoji
            if ret == True:

                img_crop = frame[220:350, 262:398]

                # transformacija iz bgr u rgb posto openCV ucitava kao bgr
                frejm = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)

                frejm_grey = cv2.cvtColor(frejm, cv2.COLOR_RGB2GRAY)

                # retww je izracunata vrednost praga, image_bin je binarna slika
                retww, frejm_bin = cv2.threshold(frejm_grey, 48, 255, cv2.THRESH_BINARY_INV)

                kernel = np.ones((3, 3))  # strukturni element 3x3 blok
                frejm_er = cv2.erode(frejm_bin, kernel, iterations=4)

                im, contours, hierarchy = cv2.findContours(frejm_er, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                #print("Number of objects found = ", len(contours)-1)
                konture = cv2.drawContours(frejm_er, contours, -1, (0, 255, 0), 3)

                #cv2.imshow('Frame', konture)

                ljudi = []
                for contour in contours:
                    ljudi.append(contour)

                ukupno = ukupno + len(ljudi)-1
                #print("Number of people for video NUMMMMMMMMMMMMMMBERRRRRRRRR= ", ukupno)

                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            # Break the loop
            else:
                break

    file.write('video' + str(video) + '.mp4,' + str(ukupno) + '\n')

file.close()

# When everything done, release the video capture object
snimak.release()

# Closes all the frames
cv2.destroyAllWindows()