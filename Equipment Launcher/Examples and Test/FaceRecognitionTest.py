import cv2
import os
import numpy as np



face_cascade = cv2.CascadeClassifier('Equipment Launcher/Haar Cascade Classifiers/haarcascade_frontalface_default.xml')
# Initialize the LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Create a label array that holds all of the labels of the faces in sync with the faces
labels = []

def addToDataset(faces, face_img, face_copy) -> None:
    for i, (x,y,w,h) in enumerate(faces):
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,0,0),2)
        cropped_image = face_copy[y: y + h, x: x + w]

        fileName = f"face_{i+1}.jpg"
        outputPath = os.path.join("datasets", fileName)

        label = f"face_{i+1}"
        labels.append(label)

        cv2.imwrite(outputPath, cropped_image)


# Just from using loaded images
def main() -> None:
    face_img = cv2.imread("Equipment Launcher/testFaces/faces-3598245545.jpeg")
    face_copy = face_img.copy()
    # face_img = cv2.resize(face_img,(0, 0), fx=0.5, fy=0.5)

    greyImg = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(greyImg, 1.1, 4)

    addToDataset(faces, face_img, face_copy)

    # Train the recognizer
    # recognizer.train(np.asarray(faces), np.asarray(labels))
    # recognizer.save('./recognizer/trained_model.yml')  # replace with your save path


    # print(faces)
    print(labels)
    # print(enumerate(faces))

    # Trying to generate individual windows for each face; it actually works
    # for face in faces:
    #     x,y,w,h = face
    #     cv2.imshow('Face #'+str(face[0]),face_img[y:y+h,x:x+w])
    #     cv2.waitKey(0)


    cv2.imshow('Face', face_img)
    cv2.waitKey(0)

# Using a webcam; not working
# def main() -> None:
#     cap = cv2.VideoCapture(0)
#
#     if not cap.isOpened():
#         print("Could not open camera")
#         return
#
#     while True:
#         ret, frame = cap.read()
#
#         greyImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(greyImg, 1.1, 4)
#
#         for (x,y,w,h) in faces:
#             cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
#
#         if not ret:
#             print("Could not read frame")
#             break
#
#         cv2.imshow("Webcam", frame)
#
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()


if __name__ == '__main__':
    main()