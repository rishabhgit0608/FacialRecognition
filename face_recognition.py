import numpy as np 
import cv2 
import os 
from sklearn.neighbors import KNeighborsClassifier

all_files=os.listdir('face_dataset')
print(all_files)

labels=[]
for file in all_files:
    if(file.endswith('.npy')):
        labels.append(file[:-4])
    else:
        continue
print(labels)

mugshots=[]
for file in all_files:
    f=np.load('face_dataset/'+file)
    mugshots.append(f)

print(len(mugshots))
#print(mugshots)
mugshots=np.concatenate(mugshots,axis=0)
print(mugshots.shape)


mugshots=mugshots.reshape((mugshots.shape[0],-1))
print(mugshots.shape)

labels=np.repeat(labels,50)
print(labels.shape)
labels=labels.reshape(labels.shape[0],-1)
print(labels.shape)

dataset=np.hstack((mugshots,labels))
print("Dataset Shape ",dataset.shape)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(dataset[:,:-1],dataset[:,-1])

cam=cv2.VideoCapture(0)
classifier=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
    ret, frame=cam.read()
    if not ret:
        continue
    faces=classifier.detectMultiScale(frame,1.3,5)
     
    for face in faces:
        x,y,w,h = face # Tuple Unpacking
        cropped_face = frame[y:y+h, x:x+w]
        cropped_face = cv2.resize(cropped_face, (100,100))
        cropped_face = cropped_face.reshape((1,-1))	
        preds = knn.predict(cropped_face)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2) # Drawing a box around the face
        cv2.putText(frame, preds[0], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
    cv2.imshow("Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

