import cv2
import numpy as np
import serial
import time

face_cascade = cv2.CascadeClassifier('HalloweenTurret/haarcascade_frontalface_default.xml')

arduinoData = serial.Serial('/dev/ttyACM0', 9600, timeout=0.01, write_timeout=0.01)

LAST_SEND = 0
SEND_INTERVAL = 0.10   # only 10 msgs/sec

def send_to_arduino(x, y):
    global LAST_SEND
    now = time.time()

    if now - LAST_SEND < SEND_INTERVAL:
        return

    data = f"{x},{y}\r"

    try:
        arduinoData.write(data.encode())
    except serial.SerialTimeoutException:
        print("Serial timeout - Arduino overloaded")
    except Exception as e:
        print("Serial error:", e)

    LAST_SEND = now


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera error")
        return
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(grey, 1.15, 6, minSize=(120,120))

        if len(faces) > 0:
            # track largest face only
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x,y,w,h = faces[0]
            forehead = int(h/3)

            # Face Box
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

            # Forehead Box and Label
            cv2.rectangle(frame, (x,y), (x+w, y+forehead), (255,0,0), 2)
            cv2.putText(frame, "Forehead", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

            send_to_arduino(x, y)

        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
