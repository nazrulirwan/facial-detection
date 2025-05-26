import cv2
import time

#facial recognition structure
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#initialize webcam
video = cv2.VideoCapture(0)
time.sleep(3)

#loop frames over camera
while video.isOpened():
    #read frame
    success, frame = video.read()
    if not success:
        break

    #flip frame(inverted camera)
    frame = cv2.flip(frame, 1)

    #convert frame to gray
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #search face
    face = face_cascade.detectMultiScale(gray_frame, 1.1, 10)

    #draw rectangle
    for (x, y, width, height) in face:
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0,0,255), 5)

    #show frame
    cv2.imshow("Facial recognition", frame)

    #close window if ESC key is pressed
    if cv2.waitKey(10) == 27:
        break