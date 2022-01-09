# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 01:41:31 2021

@author: pandy
"""

#note before running the deployment part
#try to keep all folders and file  in the same directory in which python enviroment is working on
#please change the path accordingly 
#please change the path accordingly 
#templates folder should be in the same directory in which python is executing please make sure neither your code will not work
#try to keep app.py file also in the same directory (python current directory)
#pretrained face detection model file is provided in folder saved_model please specify the path for both assigning net named model(at 46th line )

from flask import Flask, render_template, Response, request
import cv2
import face_recognition
import datetime, time
import os, sys
import numpy as np
from threading import Thread
from numpy import load
from numpy import savetxt
from numpy import save
from numpy import loadtxt
import numpy

#Please specify the path where all the project code and saved_model folder and templates folder are saved and choose same as a python directory

path='D:/facerecognitionproject/Projectcode'

#***************************************************************************************************************************************************************************************************************************************************************
global capture,rec_frame, grey, switch, upload, face, rec, out , name
capture=0
grey=0
upload=0
face=0
switch=1
rec=0


#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass




#Load pretrained face detection model    
net = cv2.dnn.readNetFromCaffe('{}/saved_model/deploy.prototxt.txt'.format(path), '{}/saved_model/res10_300x300_ssd_iter_140000.caffemodel'.format(path))

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')


#***********************************************************************************************************************************************************************************************************************************************************************

# Load a sample picture and learn how to recognize it.
#loading encoding

known_face_encodings =load('{}/encoding.npy'.format(path))
#get the names of images 
known_face_names = loadtxt('{}/imgnames.csv'.format(path),dtype='str', delimiter=',',)
known_face_names=known_face_names.tolist()

# Create arrays of known face encodings and their names

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

#****************************************************************************************************************************************************************************************************************************************************************************

camera = cv2.VideoCapture(0)

def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)
        
def classNames():
    paths = '{}/shots'.format(path)
    images = []
    classNames =[]
    myList = os.listdir(paths)
    for cl in myList:
        curImg = cv2.imread(f'{paths}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    return classNames

#********************************************************************************************************************************************************************************************************************************************************************************

def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))   
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:            
            return frame           

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame=frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = ( int(w * r), 480)
        frame=cv2.resize(frame,dim)
    except Exception as e:
        pass
    return frame

#*****************************************************************************************************************************************************************************************************************************************************************************************************
def gen_framed():  # generate frame by frame from camera
    global out, capture,rec_frame
    while True:
        success, frame = camera.read() 
        if success:
            if(face):                
                frame= detect_face(frame)
            if(grey):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if(capture):
                capture=0
                #now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "{}.jpg".format(name)])
                cv2.imwrite(p, frame)
                
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass
    
#**********************************************************************************************************************************************************************************************************************************************************************************************************

def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
           
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.50)
                name = "Unknown"
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)
            

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
#***********************************************************************************************************************************************************************************************************************************************************************************************************************
            
@app.route('/tool')
def index():
    return render_template('signup.html')

@app.route('/tool/video')
def video_feed():
    return Response(gen_framed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/Home')
def login():
    return render_template('home.html')


@app.route('/tool/requests',methods=['POST','GET'])
def tasks():
    global switch,camera,name
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            name = request.form.get('name')
            print(name)
            global capture
            capture=1
        elif  request.form.get('grey') == 'Grey':
            global grey
            grey=not grey
        elif  request.form.get('upload') == 'upload':
            global upload
        elif  request.form.get('face') == 'Face Only':
            global face
            face=not face 
            if(face):
                time.sleep(4)   
        elif  request.form.get('stop') == 'Stop/Start':
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1
                                      
    elif request.method=='GET':
        return render_template('signup.html')
    return render_template('signup.html')

@app.route('/tools')
def model():
    pathed = '{}/shots'.format(path)
    images = []
    classNames =[]
    myList = os.listdir(pathed)
    print(myList)
    for cl in myList:
        curImg = cv2.imread(f'{pathed}/{cl}')
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
    
    #*******************************************************************************************************************************
    #face encoding
     
    encodeListKnown = findEncodings(images)
    print('Encoding Complete')
    
    
    #************************************************************************************************************************************************************
    #Save Encoding of all images 
    
    # Assigning all image Encoding to variable name encoding
    encoding = encodeListKnown
    
    # save the encoding 
    save('{}/encoding.npy'.format(path), encoding) # save encodings as npy file 
    
    # save images name to csv file
    names=list(classNames)
    savetxt('{}/imgnames.csv'.format(path), names, delimiter=',',fmt ='% s') 
    return 'your image is uploaded'
    
if __name__ == '__main__':
    app.run()
