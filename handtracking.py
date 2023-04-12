import cv2 as cv
import mediapipe as mp
import time

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0


cap = cv.VideoCapture(0)
while True:
    success , img = cap.read()
    imgGRB = cv.cvtColor(img , cv.COLOR_BGR2RGBA)
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for  handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)
            
        
    cTime = time.time()
    fps = 1/(cTime- pTime)
    pTime = cTime

    cv.putText(img , str(int(fps)), (10,70),cv.FONT_HERSHEY_DUPLEX,3,(255,0,255),3)


    cv.imshow('Image ' , img)
    cv.waitKey(1)
