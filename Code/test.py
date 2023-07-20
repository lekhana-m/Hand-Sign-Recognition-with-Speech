import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import math
import time
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("model/keras_model.h5","model/labels.txt")
offset = 20
imgSize = 300

counter = 0

labels = ["A","B", "C", "D", "E","F","G","M","P","Q","S","V", "W", "Y", "Z"]
while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y - offset:y + h + offset, x-offset:x + w + offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize -wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)
            print(prediction,index)


        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)


        cv2.putText(imgOutput,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
    if (index == 0):
        alphabet = "B"
    elif (index == 1):
        alphabet = "C"
    elif (index == 2):
        alphabet = "D"
    elif (index == 3):
        alphabet = "E"
    elif (index == 4):
        alphabet = "G"
    elif (index == 5):
        alphabet = "M"
    elif (index == 6):
        alphabet = "P"
    elif (index == 7):
        alphabet = "Q"
    elif (index == 8):
        alphabet = "S"
    elif (index == 9):
        alphabet = "V"
    elif (index == 10):
        alphabet = "W"
    elif (index == 11):
        alphabet = "Y"
    elif (index == 12):
        alphabet = "Z"


    else:
        alphabet = "Show next sign"

    import pyttsx3

    text_speech = pyttsx3.init()
    text_speech.say(alphabet)
    text_speech.runAndWait()






