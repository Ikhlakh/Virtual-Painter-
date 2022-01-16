import cv2
import numpy as np
import time
import os
import Hand_Tracking_module as htm

###########
BrushThikness = 15
EraserThickess = 75
##########




folderpath = "header"
myList = os.listdir(folderpath)
print(myList)
overList = []
for imPath in myList:
    image = cv2.imread(f'{folderpath}/{imPath}')
    overList.append(image)
print(len(overList))
header =overList[0]
drawcolor =(255,0,255)

cap= cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

Detector = htm.handDetector(detectioncon=0.85)
Xp, Yp = 0, 0
imgCanvas = np.zeros((720,1280,3), np.uint8)


while True:
    # 1.import image
    success ,img = cap.read()
    img = cv2.flip(img ,1)

    # 2 .find Landmark
    img = Detector.findHands(img)
    lmList = Detector.findPosition(img ,draw=False)

    if len(lmList)  != 0:
        # print(lmList)

        # tip of index and middle finger
        x1 ,y1 =lmList[8][1:]
        x2 ,y2 = lmList[12][1:]

        # 3. Check Which Fingers are Up
        fingers = Detector.fingersup()
        # print(fingers)

        # 4.if two fingers are up - then Selection Mode
        if fingers[1] and fingers[2]:
            Xp, Yp = 0, 0
            print("Selection mode")

            # Checking for the click
            if y1 < 80:
                if 5 < x1 < 50:
                    header = overList[0]
                    drawcolor = (255, 0, 255)
                elif 300 < x1 < 400:
                    header = overList[1]
                    drawcolor = (139, 0, 0)
                elif 700 < x1 < 800:
                    header = overList[2]
                    drawcolor = (0, 255, 0)
                elif 950 < x1 < 1100:
                    header = overList[3]
                    drawcolor = (0,0,0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawcolor, cv2.FILLED)


        # 5.If Index finger is up Drawing Mode
        if fingers[1] and fingers[2]==False:
            cv2.circle(img, (x1,y1), 15 ,(255,0,255),cv2.FILLED)
            print("Drawing  mode")
            if Xp==0 and Yp==0:
                Xp, Yp = x1 , y1

            # For Eraser
            if drawcolor == (0,0,0):
                cv2.line(img ,(Xp,Yp) ,(x1,y1),drawcolor,EraserThickess)
                cv2.line(imgCanvas ,(Xp,Yp) ,(x1,y1),drawcolor,EraserThickess)
            # For Color drawing
            else:
                cv2.line(img, (Xp, Yp), (x1, y1), drawcolor, BrushThikness)
                cv2.line(imgCanvas, (Xp, Yp), (x1, y1), drawcolor, BrushThikness)

            Xp, Yp = x1, y1

    # Conver BGR to GRAY to Imginverse to bgr
    ImgGray = cv2.cvtColor(imgCanvas , cv2.COLOR_BGR2GRAY)
    _, ImgInv = cv2.threshold(ImgGray , 50 , 255,cv2.THRESH_BINARY_INV)
    ImgInv = cv2.cvtColor(ImgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,ImgInv)
    img = cv2.bitwise_or(img, imgCanvas)






    # setting header image
    img [0:125,0:1280] = header
    # fuse two images
    cv2.imshow("image", img)
    cv2.imshow("imgCanvas", imgCanvas)
    cv2.imshow("imgInv", ImgInv)
    cv2.waitKey(1)