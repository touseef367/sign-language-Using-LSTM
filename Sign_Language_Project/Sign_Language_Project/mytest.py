
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from cvzone.ClassificationModule import Classifier
import time
cap=cv2.VideoCapture(0)
detector=HandDetector(maxHands=2)
Classifier=Classifier("Model/keras_model.h5","Model/labels.txt")
offset=40
imgsize = 300

folder="Data/"
counter=0
labels=["excuse me","good bye","Good Morning","have something",
        "hellow","how are you","I have to go","i like your words","what is your name","nice to meet you",
        "No","Ok","Sorry","thank you","Wel come","Whats up","Yes"]



while True:
    success, img = cap.read()
    imgoutput=img.copy()
    hands, img = detector.findHands(img)
    try:
        if hands:
            hand=hands[0]
            x,y,w,h = hand['bbox']
            imgwhite = np.ones((imgsize,imgsize,3),np.uint8)*255
            imgCrop=img[y-offset:y+h+offset,x-offset:x+w+offset]
            imgCropshape = imgCrop.shape



            aspectratio = h/w

            if aspectratio>1:
                k = imgsize/h
                wcal=math.ceil(k*w)
                imgresize=cv2.resize(imgCrop,(wcal,imgsize))
                imgresizeshape = imgresize.shape
                wgap=math.ceil((imgsize-wcal)/2)
                imgwhite[:, wgap:wcal+wgap] = imgresize
                prediction,index = Classifier.getPrediction(imgwhite,draw=False)
                print(prediction,index)

            else:
                k = imgsize / w
                hcal = math.ceil(k * h)
                imgresize = cv2.resize(imgCrop, (imgsize, hcal))
                imgresizeshape = imgresize.shape
                hgap = math.ceil((imgsize - hcal) / 2)
                imgwhite[hgap:hcal + hgap,:] = imgresize
                prediction, index = Classifier.getPrediction(imgwhite,draw=False)

            cv2.rectangle(imgoutput, (x - offset, y - offset-50), (x - offset+320,y - offset-50+50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgoutput,labels[index],(x,y-30),cv2.FONT_HERSHEY_COMPLEX,1.7,(255,255,255),2)
            cv2.rectangle(imgoutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),4)


            cv2.imshow("imgCrop",imgCrop)
            cv2.imshow("imgwhite", imgwhite)

    except:
        pass

    cv2.imshow("image",imgoutput)
    cv2.waitKey(1)

