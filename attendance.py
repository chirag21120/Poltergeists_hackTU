import re
import face_recognition 
import cv2
import numpy as np
import os
from datetime import datetime
from datetime import date
path = 'photos'
images = []
classNames = []
myList =  os.listdir(path)
#print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    print(classNames)

    def findencodings(images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
             
        return encodeList

def markAttendance(name): 
    with open("Attendance.csv",'r+') as f:
        att = f.readlines()
        list = []
        for line in att:
            entry = line.split(',')
            list.append(entry[0])
        if name not in list:
            now = datetime.now()
            stime = now.strftime('%H:%M:%S')
            today = date.today()
            dt = today.strftime("%D")
            f.writelines(f'\n{dt},{name},{stime}')   


encodeListknown = findencodings(images)
print('Encoding Complete') 

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgs = cv2.resize(img,(0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)
    
    facesCurFrames =face_recognition.face_locations(imgs)
    encodesCurFrames = face_recognition.face_encodings(imgs,facesCurFrames)

    for encodeFace,faceLoc in zip(encodesCurFrames,facesCurFrames):
        matches = face_recognition.compare_faces(encodeListknown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListknown,encodeFace)
       # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if (matches[matchIndex]):
            name = classNames[matchIndex].upper()
           # print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            radius = x1-100
            cv2.circle(img,(x2-100,y2-45),radius,(0,0,255),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,355,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
         
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)


