import cv2
import numpy as np

# Initialize video capture on webcam
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

#  Using a webcam; not working
def main() -> None:
    # cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera")
        return

    while True:
        ret, frame = cap.read()

        # HSV Gathering
        hsvImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        ## Getting the Color Ranges of the HSV values
        # These are starting points, adjust as needed!
        lower_green = np.array([40, 100, 100]) # H: 40, S: 100, V: 100
        upper_green = np.array([80, 255, 255]) # H: 80, S: 255, V: 255

        # Making the Mask
        imgMask = cv2.inRange(hsvImg, lower_green, upper_green)


        # Extracting the Green
        # greenFrame = frame[:, :, 1]
        
        if not ret:
            print("Could not read frame")
            break


        # ## Finding the Highest Intensity Point (Green Extract Version)
        # # Creating a Mask
        # imgMask = cv2.inRange(greenFrame, 250,255)

        # # Filtering it and Cleaning it Up
        # imgMask = cv2.medianBlur(imgMask, 5)

        # # Creating a Hough Circle of the Laser Point
        # roughCircle = cv2.HoughCircles(imgMask,cv2.HOUGH_GRADIENT, 1, 20, 
        #                                param1=100, param2=30, minRadius=0, maxRadius=0)
    
        # # Drawing the Circle
        #     # Convert the coordinates of the circles to integers
        # if roughCircle is not None:
        #     roughCircle = np.uint16(np.around(roughCircle))
        #     # Loop over the detected circles
        #     for i in roughCircle[0, :]:
        #         # Draw the outer circle (center and radius) in green
        #         cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
        #         # Draw the center of the circle in red
        #         cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)

        

        cv2.imshow("Webcam", imgMask)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()