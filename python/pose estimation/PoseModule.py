import cv2
import mediapipe as mp
import time


class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(
                    self.mode,
                    self.upBody,
                    self.smooth,
                    min_detection_confidence=self.detectionCon,
                    min_tracking_confidence=self.trackCon
                )

    def findPose(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        
        return img
    
    def findPosition(self, img, pointToTrack, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                cx , cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if id == pointToTrack and draw:
                    cv2.circle(img, (cx, cy), 10, (255, 255, 255), cv2.FILLED)

                if pointToTrack < len(lmList):
                    print(lmList[pointToTrack])
                else:
                    print(f"Point {pointToTrack} not found")


                return lmList
