import cv2
import numpy as np
import serial
import time

# Load the Haar cascade file for face detection
face_cascade = cv2.CascadeClassifier('HalloweenTurret/Haar Cascade Classifiers/haarcascade_frontalface_default.xml')

# Setting up the serial communication to the Arduino
arduinoData = serial.Serial('/dev/ttyACM0', 9600, timeout=0.01, write_timeout=0.01)

LAST_SEND = 0
SEND_INTERVAL = 0.10   # only 10 msgs/sec

def send_coordinates_to_arduino(x, y):
    # Convert the coordinates to a string and send it to Arduino
    global LAST_SEND
    now = time.time()

    if now - LAST_SEND < SEND_INTERVAL:
        return

    coordinates = f"{x},{y}\r"

    try:
        arduinoData.write(coordinates.encode())
    except serial.SerialTimeoutException:
        print("Serial timeout - Arduino overloaded")
    except Exception as e:
        print("Serial error:", e)

    LAST_SEND = now



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
       
        # Drawing the Rectangle around the last face 
        if len(faces) > 0:
            # Get the coordinates of the last detected face
            (x, y, w, h) = faces[-1]

            # Drawing the rectangle and labeling the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Target Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
            
            send_coordinates_to_arduino(x, y)
        
        cv2.imshow('Face Detection', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()