import serial
import time
import numpy as np
import cv2

capture = cv2.VideoCapture(0) #Change the number for the camera that you are using, 0 is for the internal laptop camera, 1 is for an external webcam
face_cascade = cv2.CascadeClassifier('Equipment Launcher/Haar Cascade Classifiers/haarcascade_frontalface_default.xml')

while True:
    isTrue, frame = capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.05, 8, minSize=(120,120))
    if len(faces) > 0:
        # Get the coordinates of the last detected face
        (x, y, w, h) = faces[-1]

        # Find the rough position of the forehead
        forehead = h//3

        # Box the entire head
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 5)

        # Forehead Box and Label
        cv2.rectangle(frame, (x,y), (x+w, y+forehead), (255,0,0), 2)
        cv2.putText(frame, "Forehead", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        # Laser Simulation
        x_center = x+w // 2
        y_center = y+forehead // 4

        cv2.circle(frame, (x_center, y_center), 5, (0, 255, 0), -1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()