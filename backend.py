import os
import cv2
import numpy as np
import cvzone
from cvzone.PoseModule import PoseDetector

def run_backend():
    ClothsPath = "Assest/Clothes"
    clothesList = os.listdir(ClothsPath)
    print(clothesList)
    fxedRatio = 267 / 180  # width of shirt / width of pt 11 to pt 12
    clothesRatioHeightWidth = 1200 / 720
    # Read the button images
    BtnRight = cv2.imread("Assest/Buttons/R_arrow.png", cv2.IMREAD_UNCHANGED)
    BtnLeft = cv2.flip(BtnRight, 1)

    # Convert button images to RGBA format
    BtnRight = cv2.cvtColor(BtnRight, cv2.COLOR_BGR2BGRA)
    BtnLeft = cv2.cvtColor(BtnLeft, cv2.COLOR_BGR2BGRA)

    print('Shape of Button Right : ', BtnRight.shape)

    ImgNum = 0
    ChangeToRight = 0
    ChangeToLeft = 0

    # Load all images from the directory
    resized_images = []
    for img_name in clothesList:
        img_path = os.path.join(ClothsPath, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (100, 100))  # Resize images to fit the carousel
        resized_images.append(img)

    # Create a black canvas for the bottom carousel
    carousel_bottom = np.zeros((100, 100 * len(clothesList), 3), dtype=np.uint8)

    # Paste each resized image onto the bottom carousel
    x_offset = 0
    for img in resized_images:
        # Convert the image to RGB color space
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        carousel_bottom[0:img.shape[0], x_offset:x_offset + img.shape[1]] = img_rgb
        x_offset += img.shape[1]

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

    # Create a named window and set it to full screen
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        success, img = cam.read()

        img = detector.findPose(img)

        # Find the landmarks, bounding box, and center of the body in the frame
        # Set draw=True to draw the landmarks and bounding box on the image
        lmList, bboxInfo = detector.findPosition(img, draw=False, bboxWithHands=False, )

        # Check if anybody landmarks are detected
        if lmList:
            lm11 = lmList[11][0:2]
            lm12 = lmList[12][0:2]
            lm16 = lmList[16][0:2]
            lm23 = lmList[23][0:2]
            lm24 = lmList[24][0:2]

            # Calculate the real-time difference between lm pt 11 and lm pt 12
            width_lm_11_to_12 = lm12[0] - lm11[0]
            # Calculate the real-time difference between lm pt 23 and lm pt 24
            width_lm_23_to_24 = lm24[0] - lm23[0]

            imgClothes = cv2.imread(os.path.join(ClothsPath, clothesList[ImgNum]), cv2.IMREAD_UNCHANGED)

            widthOfClothes = int((lm11[0] - lm12[0]) * fxedRatio)

            imgClothes = cv2.resize(imgClothes, (widthOfClothes, int(widthOfClothes * clothesRatioHeightWidth)))

            # Specify the number of pixels to move the clothes upwards
            pixels_to_move_upwards = 100  # You can change this value as per your requirement
            pixels_to_move_right = -70  # moving little bit on the left side

            # Get window dimensions
            window_w = img.shape[1]  # Width of the current window
            window_h = img.shape[0]
            # Height of the current window

            # Calculate target button coordinates relative to window size
            target_x = 0.85 * window_w  # Right side of the window, with a margin
            target_y = 0.15 * window_h  # Top of the window, with a margin

            target_x_left = 0.03 * window_w  # Left side of the window, with a margin
            target_y_left = 0.15 * window_h  # Top of the window, with a margin

            try:
                # Adjust the y-coordinate of lm12 to move the clothes upwards
                lm12_adjusted = (lm12[0] + pixels_to_move_right, lm12[1] - pixels_to_move_upwards)
                img = cvzone.overlayPNG(img, imgClothes, lm12_adjusted)

            except:
                pass

            img = cvzone.overlayPNG(img, BtnRight, (int(target_x), int(target_y)))
            img = cvzone.overlayPNG(img, BtnLeft, (int(target_x_left), int(target_y_left)))

            frame_interval = 1 / 30  # Interval between frames (assuming 300 fps)

            if lmList[16][1] < 300:
                ChangeToRight += 1
                # Calculate the remaining time
                remaining_time = max(0, 1 - ChangeToRight * frame_interval)  # Ensure the time is not negative
                text = f"Time: {remaining_time:.1f} sec"  # Format the time as a floating-point number with one decimal place
                # Display the countdown timer
                cv2.putText(img, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

                if remaining_time == 0:  # If countdown ends
                    ImgNum += 1  # Increase ImgNum by 1
                    ChangeToRight = 0  # Reset the countdown
                    if ImgNum < len(clothesList) - 1:
                        ImgNum += 1

            elif lmList[15][0] > 950:  # Use the left wrist landmark (lmList[15]) for detecting leftward movement
                ChangeToLeft += 1
                # Calculate the remaining time
                remaining_time = max(0, 2 - ChangeToLeft * frame_interval)  # Ensure the time is not negative
                text = f"Time: {remaining_time:.1f} sec"  # Format the time as a floating-point number with one decimal place
                # Display the countdown timer
                cv2.putText(img, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

                if remaining_time == 0:  # If countdown ends
                    ImgNum -= 1  # Decrease ImgNum by 1
                    ChangeToLeft = 0  # Reset the countdown
                    if ImgNum > 0:
                        ImgNum -= 1

            else:
                ChangeToRight = 0
                ChangeToLeft = 0

        try:
            # Display the bottom carousel at the bottom of the screen
            h, w, _ = img.shape
            bottom_height, bottom_width, _ = carousel_bottom.shape
            img[h-bottom_height: h, 0: bottom_width] = carousel_bottom

            # Display the image with the overlaid clothes
            cv2.imshow("Image", img)

            # Check for key press
            key = cv2.waitKey(1)

            # If 'q' is pressed, break the loop
            if key == ord('q'):
                break

        except Exception as e:
            print("Error occurred:", e)

    # Release the camera and close all windows
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_backend()