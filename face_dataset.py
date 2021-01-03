import cv2
import numpy as np

name=input("Enter the name of the person ")
num_imgs=(int)(input("Enter the number of images "))

cam=cv2.VideoCapture(0)
classifier=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

mugshots=[]
while num_imgs:
    ret,frame=cam.read()
    if not ret:
        continue
    faces=classifier.detectMultiScale(frame,1.3,5)
    faces=sorted(faces, key=lambda e:e[2]*e[3],reverse=True)
    if not faces:
        continue
    faces=[faces[0]]

    for face in faces:
        x,y,w,h=face
        cropped_image=frame[y:y+h,x:x+w]
        cropped_image=cv2.resize(cropped_image,(100,100))
        mugshots.append(cropped_image)
        num_imgs=num_imgs-1


mugshots=np.array(mugshots)
print(mugshots.shape)
np.save('face_dataset/'+name,mugshots)
cam.release()
cv2.destroyAllWindows()
