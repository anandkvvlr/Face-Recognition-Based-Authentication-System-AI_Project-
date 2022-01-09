# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 14:23:10 2021

@author:Neelesh pandya
"""

#please change the path accordingly

#importing libraries
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from PIL import ImageGrab
import keyboard
from numpy import load
from numpy import savetxt
from numpy import save
from numpy import loadtxt

#********************************************************************************************************************************

#Getting Names of the images
 
path = 'D:/JPG'
images = []
classNames =[]
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

#**********************************************************************************************************************************
#defining function for encoding of all images

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]#72 80 104 139 146
        encodeList.append(encode)
    return encodeList

#****************************************************************************************************************************
#this is for marking attendence 
def markAttendance(name):
    with open('D:/Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'n{name},{dtString}')
            
#******************************************************************************************************************************
#FOR CAPTURING SCREEN RATHER THAN WEBCAM
def captureScreen(bbox=(300,300,690+300,530+300)):
    capScr = np.array(ImageGrab.grab(bbox))
    capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
    return capScr

#*******************************************************************************************************************************
#face encoding
 
encodeListKnown = findEncodings(images)
print('Encoding Complete')


#************************************************************************************************************************************************************
#Save Encoding of all images 

# Assigning all image Encoding to variable name encoding
encoding = encodeListKnown

# save the encoding 
save('D:/face recognition project/encoding.npy', encoding) # save encodings as npy file 

# save images name to csv file
names=list(classNames)
savetxt('D:/face recognition project/imgnames.csv', names, delimiter=',',fmt ='% s') 




#*************************************************************************************************************************************************************
# #loading the saved image this will only work if there is no change in directory where the image is stored
# import cv2
# import numpy as np
# import face_recognition
# import os
# from datetime import datetime
# from PIL import ImageGrab
# import keyboard
# from numpy import load
# from numpy import savetxt
# from numpy import save
# from numpy import loadtxt

# #loading encoding

# encodeListKnown =load('D:/face recognition project/encoding.npy')
# #get the names of images 
# classNames = loadtxt('D:/face recognition project/imgnames.csv',dtype='str', delimiter=',',)

#***********************************************************************************************************************************************************************
#capture the image from webcam for face recognition   

     
print("[INFO] Video Capture is now starting please stay still")
        
cap = cv2.VideoCapture(0)
 
while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace,tolerance=0.50)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]==False:
            name='Unknown'
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            # Show the image
            cv2.imshow("Identified Face", img)
            key = cv2.waitKey(1) & 0xFF
            print('your face has  not  been registered To register press q and then s')
            key = cv2.waitKey(1) & 0xFF
            print('your face has been registered')
            if key == ord('s'):
                cap.release()
                cv2.destroyAllWindows()
        elif matches[matchIndex]==True:
            name = classNames[matchIndex].upper()
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            #markAttendance(name)
            cv2.imshow("Identified Face", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()



