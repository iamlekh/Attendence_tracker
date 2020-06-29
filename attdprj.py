import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
path = 'img_attd'
images = []
classNames = []
mylist = os.listdir(path)
print(mylist)


for cl in mylist:
    curImage = cv2.imread(f'{path}/{cl}' )
    images.append(curImage)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findencoding(images):
    encodedlist = []
    for img in images:
        cv2.cvtColor(img ,cv2.COLOR_BGR2RGB)
        encoded = face_recognition.face_encodings(img)[0]
        encodedlist.append(encoded)
    return encodedlist

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList =[]
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in  line:
            now = datetime.now()
            dt_string = now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dt_string}')



encodedknownface = findencoding(images)
print("encoding_completed...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodedknownface, encodeFace)
        faceDis = face_recognition.face_distance(encodedknownface, encodeFace)
        # print(faceDis)
        matchidx = np.argmin(faceDis)

        if matches[matchidx]:
            name = classNames[matchidx].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1, y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1, y1 -35),(x2,y2),(0,255,0),2)
            cv2.putText(img, name, (x1 +6,y1-6), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255),1)
            markAttendance(name)
    cv2.imshow('Input', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print("end")
