import cv2
import time
import face_recognition
import numpy as np
import os
from os import listdir

#facial recognition structure
#face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#initialize webcam
video = cv2.VideoCapture(0)
time.sleep(3)

#load jpgs folder and get encodings and names
known_encodings = []
known_names = []
folder_path = "jpgs"

for filename in os.listdir(folder_path):
    if filename.lower().endswith(".jpg"):

        file_path = os.path.join(folder_path, filename)

        image = face_recognition.load_image_file(file_path)
        image_encoding = face_recognition.face_encodings(image)
        if not image_encoding:
            raise RuntimeError("No face found")
        else:
            known_encodings.append(image_encoding[0])
            known_names.append(os.path.splitext(filename)[0])
    else:
        print("no .jpg files")

#loop frames over camera
while video.isOpened():
    #read frame
    success, frame = video.read()
    if not success:
        break

    #flip frame(inverted camera)
    frame = cv2.flip(frame, 1)

    #convert frame to rgb
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #detect faces in frame and get encodings
    face_location = face_recognition.face_locations(rgb_frame)
    face_encoding = face_recognition.face_encodings(rgb_frame, face_location)

    #compare face encoding in frame to known encoding in system
    for (top, right, bottom, left), face in zip(face_location, face_encoding):
        
        matches = face_recognition.compare_faces(known_encodings, face)
        face_distance = face_recognition.face_distance(known_encodings, face)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_names[best_match_index]
            color = (0, 255, 0)#green
        else:
            name = "Unknown"
            color = (0, 0, 255)#red

    #draw rectangle
    #for top, right, bottom, left in face_location:
        cv2.rectangle(frame, (left, top), (right, bottom), color, 5)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_COMPLEX, 1, color, 1)

    #show frame
    cv2.imshow("Facial recognition", frame)

    #close window if ESC key is pressed
    if cv2.waitKey(10) == 27:
        break