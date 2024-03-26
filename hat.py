import os
import cv2
import numpy as np
from cvzone.PoseModule import PoseDetector

def overlay_hat(img, lmList, hat_image_path):
    if len(lmList) >= 6:  # Ensure lmList contains at least 6 landmarks (including left and right eyes)
        # Load hat image with alpha channel
        hat_image = cv2.imread(hat_image_path, cv2.IMREAD_UNCHANGED)

        # Get landmarks for left and right eyes
        left_eye = lmList[2][0:2]  # Landmark for left eye (index 2)
        right_eye = lmList[5][0:2]  # Landmark for right eye (index 5)

        # Calculate the position to overlay the hat
        hat_x = min(left_eye[0], right_eye[0]) - 120  # Set hat's x-coordinate to the minimum x-coordinate of the eyes
        hat_y = min(left_eye[1], right_eye[1]) - 190  # Move the hat 50 pixels above the minimum y-coordinate of the eyes

        # Overlay hat on the face
        for i in range(hat_image.shape[0]):
            for j in range(hat_image.shape[1]):
                if hat_image[i, j, 3] > 0:  # Check alpha channel for transparency
                    img[int(hat_y) + i, int(hat_x) + j] = hat_image[i, j, :3]

    return img

def run_backend():
    # Initialize camera
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 2080)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 2000)

    detector = PoseDetector(staticMode=False,
                            modelComplexity=1,
                            smoothLandmarks=True,
                            enableSegmentation=False,
                            smoothSegmentation=True,
                            detectionCon=0.5,
                            trackCon=0.5)

    hat_image_path = "Assest/Hat/grad-removebg-preview.png"

    while True:
        success, img = cam.read()

        img = detector.findPose(img)

        # Find the landmarks in the image
        lmList, _ = detector.findPosition(img, draw=False)

        # Overlay hat
        if lmList and len(lmList) >= 6:  # Ensure lmList contains at least 6 landmarks (including left and right eyes)
            img = overlay_hat(img, lmList, hat_image_path)

        # Display the resulting frame
        cv2.imshow("AR Effects", img)

        # Check for key press
        key = cv2.waitKey(1)

        # If 'q' is pressed, break the loop
        if key == ord('q'):
            break

    # Release the camera and close all windows
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_backend()
