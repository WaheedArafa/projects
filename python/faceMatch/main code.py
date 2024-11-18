import cv2
from simple_facerec import SimpleFacerec


sfr = SimpleFacerec()
#load images from a folder and the app use these photos to compare with the face in the camera
sfr.load_encoding_images("imgs")  

#open the camera
cap = cv2.VideoCapture(0)

#compare camera frames with photos in the folder
while True:
    #read the frames
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
    
    #compare faces and give you the name of the face in the frame
    face_locations, face_names = sfr.detect_known_faces(frame)

    #put a square around the face and put the name of the face
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    #show the frame
    cv2.imshow("Frame", frame)

    #end app
    key = cv2.waitKey(1)
    if key == 32:
        break

#close camera and windows
cap.release()
cv2.destroyAllWindows()
