import FaceDetectionModule as fdm
cap = fdm.cv2.VideoCapture('videos/2.mp4')
pTime = 0

detector = fdm.FaceDetector()
while True:
        success, img = cap.read()
        img, bboxs = detector.findFaces(img)

        print(bboxs)

        cTime = fdm.time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        fdm.cv2.putText(img, str(int(fps)),(10,70),fdm.cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)

        fdm.cv2.imshow("Image", img)

        key = fdm.cv2. waitKey(10)
        if key == 27:
            break