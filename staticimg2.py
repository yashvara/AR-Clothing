import cv2
import mediapipe as mp
import math
import numpy as np
import matplotlib.pyplot as plt
from rembg import remove
import index


class PoseDetector:
    def __init__(self, staticMode=False, modelComplexity=1, smoothLandmarks=True, enableSegmentation=False,
                 smoothSegmentation=True, detectionCon=0.5, trackCon=0.5):
        self.staticMode = staticMode
        self.modelComplexity = modelComplexity
        self.smoothLandmarks = smoothLandmarks
        self.enableSegmentation = enableSegmentation
        self.smoothSegmentation = smoothSegmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.staticMode,
                                     model_complexity=self.modelComplexity,
                                     smooth_landmarks=self.smoothLandmarks,
                                     enable_segmentation=self.enableSegmentation,
                                     smooth_segmentation=self.smoothSegmentation,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findSegmentation(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if np.any(self.results.segmentation_mask):
            return cv2.cvtColor(self.results.segmentation_mask, cv2.COLOR_GRAY2BGR)
        return None

    def findPosition(self, img, draw=True, bboxWithHands=False):
        self.lmList = []
        self.bboxInfo = {}
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                self.lmList.append([cx, cy, cz])

            ad = abs(self.lmList[12][0] - self.lmList[11][0]) // 2
            if bboxWithHands:
                x1 = self.lmList[16][0] - ad
                x2 = self.lmList[15][0] + ad
            else:
                x1 = self.lmList[12][0] - ad
                x2 = self.lmList[11][0] + ad

            y2 = self.lmList[29][1] + ad
            y1 = self.lmList[1][1] - ad
            bbox = (x1, y1, x2 - x1, y2 - y1)
            cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + bbox[3] // 2

            self.bboxInfo = {"bbox": bbox, "center": (cx, cy)}

            if draw:
                cv2.rectangle(img, bbox, (255, 0, 255), 3)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return self.lmList, self.bboxInfo

    def findDistance(self, p1, p2, img=None, color=(255, 0, 255), scale=5):
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)

        if img is not None:
            cv2.line(img, (x1, y1), (x2, y2), color, max(1, scale // 3))
            cv2.circle(img, (x1, y1), scale, color, cv2.FILLED)
            cv2.circle(img, (x2, y2), scale, color, cv2.FILLED)
            cv2.circle(img, (cx, cy), scale, color, cv2.FILLED)

        return length, img, info

    def findAngle(self, p1, p2, p3, img=None, color=(255, 0, 255), scale=5):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        if img is not None:
            cv2.line(img, (x1, y1), (x2, y2), color, max(1, scale // 3))
            cv2.circle(img, (x1, y1), scale, color, cv2.FILLED)
            cv2.circle(img, (x2, y2), scale, color, cv2.FILLED)
            cv2.circle(img, (x3, y3), scale, color, cv2.FILLED)

        return angle, img, (x1, y1, x2, y2, x3, y3)

    def overlay_shirt(self, img, shirt_img):
        self.findPose(img, draw=False)
        lmList, _ = self.findPosition(img, draw=False)

        left_shoulder = self.lmList[11]
        right_shoulder = self.lmList[12]
        left_hip = self.lmList[23]
        right_hip = self.lmList[24]

        body_width = int(math.hypot(right_shoulder[0] - left_shoulder[0], right_shoulder[1] - left_shoulder[1]))
        body_height = int(math.hypot(right_shoulder[0] - right_hip[0], right_shoulder[1] - right_hip[1]))
        resized_shirt_img = cv2.resize(shirt_img, (body_width, body_height))

        # Check if the resized shirt image has an alpha channel
        if len(resized_shirt_img.shape) == 3 and resized_shirt_img.shape[2] == 4:
            # Convert resized shirt image from RGBA to RGB
            resized_shirt_img = cv2.cvtColor(resized_shirt_img, cv2.COLOR_RGBA2RGB)

        # Static form the image only
        shirt_height = 1200
        shirt_width = 616
        roi = img[min(left_shoulder[1], right_shoulder[1]):min(left_shoulder[1], right_shoulder[1]) + shirt_height,
              min(left_shoulder[0], right_shoulder[0]):min(left_shoulder[0], right_shoulder[0]) + shirt_width]

        print("Shapes: roi:", roi.shape, "resized shirt:", resized_shirt_img.shape)

        if roi.shape[:2] != resized_shirt_img.shape[:2]:
            resized_shirt_img = cv2.resize(resized_shirt_img, (roi.shape[1], roi.shape[0]))

        for c in range(0, min(3, roi.shape[2])):
            if resized_shirt_img.shape[2] == 3:
                alpha = np.ones((resized_shirt_img.shape[0], resized_shirt_img.shape[1]),
                                dtype=resized_shirt_img.dtype) * 255
            else:
                alpha = resized_shirt_img[:, :, 3]

            roi[:, :, c] = roi[:, :, c] * (1 - alpha / 255.0) + resized_shirt_img[:, :, c] * (alpha / 255.0)

        img[min(left_shoulder[1], right_shoulder[1]):min(left_shoulder[1], right_shoulder[1]) + shirt_height,
        min(left_shoulder[0], right_shoulder[0]):min(left_shoulder[0], right_shoulder[0]) + shirt_width] = roi

        cv2.imshow('Overlay Result', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return img

    def overlayImage(self, img, overlayImg, position):
        scale_factor = 1.7
        # Calculate the new width and height
        width = int((position[1] - position[0]) * scale_factor)
        height = int((position[2] - position[0]) * scale_factor)

        # Resize the overlay image
        overlayImg_resized = cv2.resize(overlayImg, (width, height))

        # Calculate the new position for the overlay image
        x = position[3] - (width - (position[3] - position[1])) // 2
        y = position[1] - (height - (position[3] - position[1])) // 2

        x = max(0, min(x, img.shape[1] - width - 8))
        y = max(0, min(y, img.shape[0] - height))

        # Extract the alpha channel from the overlay image
        alpha = overlayImg_resized[:, :, 3] / 255.0

        # Blend the overlay image onto the input image using alpha blending
        for c in range(0, min(3, img.shape[2])):
            img[y:y + height, x:x + width, c] = \
                img[y:y + height, x:x + width, c] * (1 - alpha) + overlayImg_resized[:, :, c] * alpha

        return img

    def segmentClothing(self, img):
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_clothes_color = np.array([0, 40, 40])
        upper_clothes_color = np.array([30, 255, 255])
        clothes_mask = cv2.inRange(hsv_image, lower_clothes_color, upper_clothes_color)
        clothes_only = cv2.bitwise_and(img, img, mask=clothes_mask)
        return clothes_only

    def overlayTshirt(self, img, tshirt_img, position):
        # Calculate the width and height based on the position
        width = position[3] - position[1] + 30
        height = position[3] - position[1] + 40

        # Resize the t-shirt image to match the calculated width and height
        tshirt_img = cv2.resize(tshirt_img, (width, height))

        # Calculate the coordinates for overlaying the t-shirt with a leftward shift
        x_offset = 30
        y_offset = 23
        x = max(0, position[0] - x_offset)
        y = max(0, position[1] - y_offset)

        # Ensure the coordinates are within the image boundaries
        x = max(0, min(x, img.shape[1] - width))
        y = max(0, min(y, img.shape[0] - height))

        # Convert to RGBA if necessary
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        if len(tshirt_img.shape) == 2 or tshirt_img.shape[2] == 1:
            tshirt_img = cv2.cvtColor(tshirt_img, cv2.COLOR_GRAY2BGRA)
        elif tshirt_img.shape[2] == 3:
            tshirt_img = cv2.cvtColor(tshirt_img, cv2.COLOR_BGR2BGRA)

        # Create a mask from the t-shirt image's alpha channel
        mask = tshirt_img[:, :, 3] / 255.0

        # Resize the mask to match the dimensions of the t-shirt image
        mask_resized = cv2.resize(mask, (tshirt_img.shape[1], tshirt_img.shape[0]))

        # Invert the mask
        mask_inv = 1.0 - mask_resized

        # Region of interest (ROI) on the original image
        img_roi = img[y:y + height, x:x + width]

        # Take only the t-shirt region from the t-shirt image
        tshirt_fg = tshirt_img[:, :, :3]

        # Ensure the shapes are compatible for blending
        img_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGRA2BGR)
        tshirt_fg = cv2.resize(tshirt_fg, (img_roi.shape[1], img_roi.shape[0]))

        # Blend the images using numpy operations
        img_roi_with_tshirt = (img_roi * (1.0 - mask_inv[:, :, np.newaxis])) + (
                tshirt_fg * mask_resized[:, :, np.newaxis])

        # Update the original image with the overlay
        img[y:y + height, x:x + width, :3] = img_roi_with_tshirt

        return img


def add_tshirt_to_image(clothing_path):
    poseDetector = PoseDetector()
    img = cv2.imread('./Assest/Reference/man2.png')
    poseDetector.findPose(img)
    lmList, _ = poseDetector.findPosition(img)

    left_shoulder = lmList[11]
    right_shoulder = lmList[12]
    left_hip = lmList[23]
    right_hip = lmList[24]

    # Calculate the position of the t-shirt
    adjustment = 30
    tshirt_position = (min(left_shoulder[0], right_shoulder[0]), min(left_shoulder[1], right_shoulder[1]) - adjustment,
                       max(left_shoulder[0], right_shoulder[0]), max(left_hip[1], right_hip[1]))

    # Read the t-shirt image and remove the background
    tshirt_image = cv2.imread(clothing_path)
    tshirt_image_no_bg = remove(tshirt_image)

    # Convert t-shirt image from RGBA to RGB
    tshirt_image_no_bg = cv2.cvtColor(tshirt_image_no_bg, cv2.COLOR_RGBA2RGB)

    # Segment clothing and overlay t-shirt
    clothing_segmented = poseDetector.segmentClothing(img)
    img_with_tshirt = poseDetector.overlayTshirt(clothing_segmented, tshirt_image_no_bg, tshirt_position)
    rgb_img = cv2.cvtColor(img_with_tshirt, cv2.COLOR_BGR2RGB)
    return rgb_img