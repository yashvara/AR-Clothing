import os
import cv2
import numpy as np
import cvzone
from cvzone.PoseModule import PoseDetector
import torch
from neural_network import scaler, model, label_encoder


def find_size(wt, ht, age):
    # sequesnce : wt, age, ht
    demo_data = np.array([[wt,age,ht]])

    demo_data_scaled = scaler.transform(demo_data)

    demo_data_tensor = torch.tensor(demo_data_scaled, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(demo_data_tensor)
        _, predicted_classes = torch.max(outputs.data, 1)

    predicted_labels = label_encoder.inverse_transform(predicted_classes.numpy())
    tshirt_s = ""
    for i, label in enumerate(predicted_labels):
        # print(f"T-shirt Size : {label}")
        tshirt_s = label
    # print(tshirt_s)
    return tshirt_s

def run_backend(wt, ht, age):

    ClothsPath = "Assest/shirts"
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

    resized_images = []
    for img_name in clothesList:
        img_path = os.path.join(ClothsPath, img_name)
        img = cv2.imread(img_path)
        resized_images.append(img)

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
            lm18 = lmList[18][0:2]

            # Calculate the real-time difference between lm pt 11 and lm pt 12
            width_lm_11_to_12 = lm12[0] - lm11[0]
            # Calculate the real-time difference between lm pt 23 and lm pt 24
            width_lm_23_to_24 = lm24[0] - lm23[0]

            imgClothes = cv2.imread(os.path.join(ClothsPath, clothesList[ImgNum]), cv2.IMREAD_UNCHANGED)

            widthOfClothes = int((lm11[0] - lm12[0]) * fxedRatio)

            imgClothes = cv2.resize(imgClothes, (widthOfClothes, int(widthOfClothes * clothesRatioHeightWidth)))

            # Specify the number of pixels to move the clothes upwards
            pixels_to_move_upwards = 130
            pixels_to_move_right = -70  # moving little bit on the left side

            # Get window dimensions
            window_w = img.shape[1]  # Width of the current window
            window_h = img.shape[0]
            # Height of the current window

            target_x = 0.85 * window_w  # Right side
            target_y = 0.15 * window_h  # Top of the window

            target_x_left = 0.03 * window_w  # Left side
            target_y_left = 0.15 * window_h  # Top of the window

            target_x_ChnageBtn = 0.03 * window_w
            target_y_ChangeBtn = 0.79 * window_h

            # ====================== display size ===============================

            t_shirt_size = find_size(wt, ht, age)
            cv2.putText(img, t_shirt_size, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)



            # =====================================================================
            try:
                lm12_adjusted = (lm12[0] + pixels_to_move_right, lm12[1] - pixels_to_move_upwards)
                img = cvzone.overlayPNG(img, imgClothes, lm12_adjusted)

            except:
                pass

            distance_to_change_cloths_btn = np.sqrt(
                (lmList[8][0] - target_x_ChnageBtn) ** 2 + (lmList[8][1] - target_y_ChangeBtn) ** 2)

            img = cvzone.overlayPNG(img, BtnRight, (int(target_x), int(target_y)))
            img = cvzone.overlayPNG(img, BtnLeft, (int(target_x_left), int(target_y_left)))

            frame_interval = 1 / 30

            if lmList[16][1] < 300:
                ChangeToRight += 1
                # Calculate the remaining time
                remaining_time = max(0, 1 - ChangeToRight * frame_interval)
                text = f"Time: {remaining_time:.1f} sec"
                # Display the countdown timer
                cv2.putText(img, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

                if remaining_time == 0:  # If countdown ends
                    ImgNum += 1  # Increase ImgNum by 1
                    ChangeToRight = 0  # Reset the countdown
                    if ImgNum < len(clothesList) - 1:
                        ImgNum += 1

            elif lmList[15][0] > 950:
                ChangeToLeft += 1
                remaining_time = max(0, 1 - ChangeToLeft * frame_interval)  # Ensure the time is not negative
                text = f"Time: {remaining_time:.1f} sec"  # Format the time as a floating-point number with one decimal place
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
            # Display the image with the overlaid clothes
            cv2.imshow("Image", img)

            # Check for key press
            key = cv2.waitKey(1)

            # If 'q' is pressed, break the loop
            if key == ord('q'):
                break

        except Exception as e:
            print("Error occurred:", e)

    cam.release()
    cv2.destroyAllWindows()

# if __name__ == "__main__":
#     run_backend()