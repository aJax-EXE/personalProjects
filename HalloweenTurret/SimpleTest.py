import cv2
import numpy as np
import serial
import time

# Load the Haar cascade file for face detection
face_cascade = cv2.CascadeClassifier('HalloweenTurret/haarcascade_frontalface_default.xml')

# Setting up the serial communication to the Arduino
arduinoData = serial.Serial('/dev/ttyACM0', 9600, write_timeout=1)

def send_coordinates_to_arduino(x, y, w, h):
    # Convert the coordinates to a string and send it to Arduino
    coordinates = f"{x},{y}\r"
    arduinoData.write(coordinates.encode())
    print(f"X{x}Y{y}\n")

def main():
    # Setting up the webcam camera feed; 0 is webcam, 1 is an external camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        # Checks if the camera feed is running or not
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(grey, 1.05, 8, minSize=(120,120))
       
        # Drawing the Rectangle and labeling the faces the different faces
        for i, (x,y,w,h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"face_{i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
            # print(f"highest number: {i+1}; x: {x}, y: {y}, w: {w}, and h: {h}")
            # largest_i = len(faces) - 1
            (lastX, lastY, lastW, lastH) = faces[-1]
            send_coordinates_to_arduino(lastX, lastY, lastW, lastH)
        
        cv2.imshow('Face Detection', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()