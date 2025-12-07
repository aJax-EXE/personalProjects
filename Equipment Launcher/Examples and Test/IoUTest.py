import cv2
import numpy as np

# Load the Haar cascade files for face detection
face_cascade_frontal = cv2.CascadeClassifier('Equipment Launcher/Haar Cascade Classifiers/haarcascade_frontalface_default.xml')
face_cascade_profile = cv2.CascadeClassifier('Equipment Launcher/Haar Cascade Classifiers/haarcascade_profileface.xml')

# Initialize video capture on webcam
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()


# Helper function to calculate IoU (Intersection over Union)
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the coordinates of the intersection rectangle
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)

    # Calculate the area of intersection
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # Calculate the area of each rectangle
    box1_area = w1 * h1
    box2_area = w2 * h2

    # Calculate IoU
    iou = inter_area / float(box1_area + box2_area - inter_area) if (box1_area + box2_area - inter_area) > 0 else 0
    return iou


# Merge detections from both frontal and profile detections
def merge_detections(detections1, detections2, iou_threshold=0.2, min_size=160):
    merged_faces = []

    # Filter small detections in detections1
    filtered_detections1 = [face for face in detections1 if face[2] >= min_size and face[3] >= min_size]

    # Process filtered detections1
    for face1 in filtered_detections1:
        keep = True
        for face2 in merged_faces:
            if calculate_iou(face1, face2) > iou_threshold:
                keep = False
                break
        if keep:
            merged_faces.append(face1)

    # Filter small detections in detections2
    filtered_detections2 = [face for face in detections2 if face[2] >= min_size and face[3] >= min_size]

    # Process filtered detections2, comparing with merged_faces
    for face2 in filtered_detections2:
        keep = True
        for face1 in merged_faces:
            if calculate_iou(face1, face2) > iou_threshold:
                keep = False
                break
        if keep:
            merged_faces.append(face2)

    return merged_faces


while True:
    ret, frame = cap.read()

    # Check if frame is captured
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame using both cascades
    faces_frontal = face_cascade_frontal.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    faces_profile = face_cascade_profile.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    # Get merged list of faces
    unique_faces = merge_detections(faces_frontal, faces_profile)
    faceNum = len(unique_faces)

    # Getting the last detected merged face
    if len(unique_faces) > 0:
        lastFace = unique_faces[len(unique_faces)-1]
        print(lastFace)
        # Drawing a rectangle and label around the last detected merged face
        for (x, y, w, h) in lastFace:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, 'Last_Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
    else:
        # Draw rectangles around merged faces and add labels
        for i, (x,y,w,h) in enumerate(unique_faces):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f'face_{i+1}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Draw rectangles around merged faces and add labels
    for i, (x,y,w,h) in enumerate(unique_faces):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f'face_{i+1}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Drawing a rectangle and label around the last detected merged face
    for (x,y,w,h) in lastFace:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, 'Last_Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)

    #seemingly works; never tested though...; don't think it works
    # for i, (x, y, w, h) in enumerate(unique_faces):
    #     cv2.rectangle(frame, (unique_faces[i-1][0], unique_faces[i-1][1]), (unique_faces[i-1][0] + unique_faces[i-1][2], unique_faces[i-1][1] + unique_faces[i-1][3]), (255, 0, 0), 2)
    #     cv2.putText(frame, 'Last_Face', (unique_faces[i-1][0], unique_faces[i-1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)

    # Count total unique faces detected
    total_faces = len(unique_faces)
    print(f"Total unique faces detected: {total_faces}")
    # print(unique_faces)

    # Display the frame with detected faces
    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()