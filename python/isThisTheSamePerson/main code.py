import cv2
import face_recognition


#load the image
img1 = cv2.imread("image bath")
#converting the color format from BGR to RGB because face recognition don't use images in BGR format
rgb_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#face encoding (it will make difference between each person so using this the program can compare faces)
img1_encoding = face_recognition.face_encodings(rgb_img1)[0]


#load the image
img2 = cv2.imread("image bath")
#converting the color format from BGR to RGB because face recognition don't use images in BGR format
rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
#face encoding (it will make difference between each person so using this the program can compare faces)
img2_encoding = face_recognition.face_encodings(rgb_img2)[0]


#comparing faces
result = face_recognition.compare_faces([img1_encoding], img2_encoding)
#the result
print("Result:", result)


#end program
exit()
