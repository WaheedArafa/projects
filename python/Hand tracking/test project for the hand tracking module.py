import cv2
import mediapipe as mp
import time
import handTrackingModule as htm

pTime = 0
cTime = 0

pointToTrack = int(input("Choose the point you want to track: "))

cap = cv2.VideoCapture(0)

detector = htm.handDetector()

while True:
    succes, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, pointToTrack)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv2.imshow("Image", img)
    key = cv2. waitKey(1)
    if key == 27:
       break
