import cv2
import datetime
from tabulate import tabulate
from simple_facerec import SimpleFacerec
import csv
import time
import mediapipe as mp

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(False, 2)
drawSpec = mpDraw.DrawingSpec(thickness=1,circle_radius=1)

pTime = 0


sfr = SimpleFacerec()
sfr.load_encoding_images("imgs") 

students = []

cap = cv2.VideoCapture('videos/2.mp4')


while True:
    success, img = cap.read()
    if not success:
       print("Error.. try again")
       break
  
    face_locations, face_names = sfr.detect_known_faces(img)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = faceMesh.process(imgRGB)
    
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
    
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        if name == "Unknown":
           break
        if name not in [student[0] for student in students]:
           students.append((name, current_time))
        
    if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)

    cv2.imshow("attendance system", img)

    key = cv2.waitKey(1)
    if key == 27:
       break

table_data = [(idx + 1, student[0], student[1]) for idx, student in enumerate(students)]


print(tabulate(table_data, headers=["No.", "Student Name", datetime.datetime.now().strftime("%Y-%m-%d")], tablefmt="grid"))

with open("attendance.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["No.", "Student Name", datetime.datetime.now().strftime("%Y-%m-%d")])
    writer.writerows(table_data)
