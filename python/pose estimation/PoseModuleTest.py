import PoseModule as PM
    
cap = PM.cv2.VideoCapture('videos/2.mp4')
pTime = 0

pointToTrack = int(input("Choose the point you want to track: "))

detector = PM.poseDetector()
while True:
        success, img = cap.read()
        img = detector.findPose(img)
        Lmlist = detector.findPosition(img, pointToTrack)

        cTime = PM.time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        PM.cv2.putText(img, str(int(fps)),(10,70),PM.cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

        PM.cv2.imshow("Image", img)
        key = PM.cv2. waitKey(1)
        if key == 27:
           break