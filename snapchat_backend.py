import os
import cv2
import numpy as np
from cvzone.PoseModule import PoseDetector

def overlay_goggles(img, lmList, goggles_image_path):
    if len(lmList) >= 6:  # Ensure lmList contains at least 6 landmarks (including left and right eyes)
        # Load goggles image with alpha channel
        goggles_image = cv2.imread(goggles_image_path, cv2.IMREAD_UNCHANGED)

        # Get landmarks for left and right eyes
        left_eye = lmList[2][0:2]  # Landmark for left eye (index 2)
        right_eye = lmList[5][0:2]  # Landmark for right eye (index 5)

        # Calculate the distance between left and right eyes
        distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))

        # Define the fixed ratio
        fixed_ratio = 240 / distance  # Fixed ratio: 240 / width of pt 2 to pt 5

        # Calculate the actual width of the specs
        actual_width = distance * fixed_ratio

        # Calculate the scaling factor to adjust the width of the goggles image
        scale_factor = actual_width / goggles_image.shape[1]  # Assuming the width of the image represents the width of the goggles

        # Resize the goggles image based on the scaling factor
        goggles_width = int(goggles_image.shape[1] * scale_factor)
        goggles_height = int(goggles_image.shape[0] * scale_factor)
        goggles_image_resized = cv2.resize(goggles_image, (goggles_width, goggles_height))

        # Calculate the position to overlay the goggles
        x = int(left_eye[0] - (goggles_width / 2))
        y = int(left_eye[1] - (goggles_height / 2))

        # Overlay goggles on the face
        for i in range(goggles_height):
            for j in range(goggles_width):
                if goggles_image_resized[i, j, 3] > 0:  # Check alpha channel for transparency
                    img[y + i, x + j] = goggles_image_resized[i, j, :3]
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

    goggles_image_path = "Assest/Specs/new_specs-removebg-preview.png"

    while True:
        success, img = cam.read()

        img = detector.findPose(img)

        # Find the landmarks in the image
        lmList, _ = detector.findPosition(img, draw=False)

        # Check if left and right eye landmarks are detected
        if lmList and len(lmList) >= 6:  # Ensure lmList contains at least 6 landmarks (including left and right eyes)
            # Overlay goggles on the face
            img = overlay_goggles(img, lmList, goggles_image_path)

        # Display the resulting frame
        cv2.imshow("AR Goggles", img)

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

