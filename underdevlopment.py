import cv2
import mediapipe as mp
import math
import numpy as np
import matplotlib.pyplot as plt

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

        if resized_shirt_img.shape[2] == 3:
            resized_shirt_img = cv2.cvtColor(resized_shirt_img, cv2.COLOR_RGB2RGBA)

        # Static form the image only
        shirt_height = 1200
        shirt_width = 616
        roi = img[min(left_shoulder[1], right_shoulder[1]):min(left_shoulder[1], right_shoulder[1]) + shirt_height,
              min(left_shoulder[0], right_shoulder[0]):min(left_shoulder[0], right_shoulder[0]) + shirt_width]

        if roi.shape != resized_shirt_img.shape:
            resized_shirt_img = cv2.resize(resized_shirt_img, (roi.shape[1], roi.shape[0]))

        for c in range(0, 3):
            roi[:, :, c] = roi[:, :, c] * (1 - resized_shirt_img[:, :, 3] / 255.0) + resized_shirt_img[:, :, c] * (
                        resized_shirt_img[:, :, 3] / 255.0)

        img[min(left_shoulder[1], right_shoulder[1]):min(left_shoulder[1], right_shoulder[1]) + shirt_height,
        min(left_shoulder[0], right_shoulder[0]):min(left_shoulder[0], right_shoulder[0]) + shirt_width] = roi

        cv2.imshow('Overlay Result', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return img

    def overlayImage(self, img, overlayImg, position):
        scale_factor = 1.4

        # Calculate the new width and height
        width = int((position[2] - position[0]) * scale_factor)
        height = int((position[3] - position[1]) * scale_factor)

        # Resize the overlay image
        overlayImg_resized = cv2.resize(overlayImg, (width, height))

        # Calculate the new position for the overlay image
        x = position[0] - (width - (position[2] - position[0])) // 2
        y = position[1] - (height - (position[3] - position[1])) // 2

        x = max(0, min(x, img.shape[1] - width))
        y = max(0, min(y, img.shape[0] - height))

        # Extract the alpha channel from the overlay image
        alpha = overlayImg_resized[:, :, 3] / 255.0

        # Blend the overlay image onto the input image using alpha blending
        for c in range(0, 3):
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

    def overlayTshirt(self, img, tshirt_img, position, width_increase=50):
        width = position[2] - position[0] + width_increase
        height = position[3] - position[1]
        tshirt_img = cv2.resize(tshirt_img, (width, height))
        x = position[0] - (width - (position[2] - position[0])) // 2
        y = position[1] - (height - (position[3] - position[1])) // 2
        x = max(0, min(x, img.shape[1] - width))
        y = max(0, min(y, img.shape[0] - height))
        img[y:y + height, x:x + width] = tshirt_img
        return img

    def make_background_white(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [cnt], 0, 255, -1)
        img[mask == 0] = [255, 255, 255]
        return img


poseDetector = PoseDetector()
img = cv2.imread('./Assest/Reference/man2.png')
poseDetector.findPose(img)
lmList, _ = poseDetector.findPosition(img)

left_shoulder = lmList[11]
right_shoulder = lmList[12]
left_hip = lmList[23]
right_hip = lmList[24]

# Calculate the position of the t-shirt
adjustment = 22
tshirt_position = (min(left_shoulder[0], right_shoulder[0]), min(left_shoulder[1], right_shoulder[1]) - adjustment,
                   max(left_shoulder[0], right_shoulder[0]), max(left_hip[1], right_hip[1]))

# Read the t-shirt image and remove the background
from rembg import remove
tshirt_path = 'Assest/Clothes/tshirt1.png'
tshirt_image = cv2.imread(tshirt_path)
tshirt_image_no_bg = remove(tshirt_image)

# Segment clothing and overlay t-shirt
clothing_segmented = poseDetector.segmentClothing(img)
img_with_tshirt = poseDetector.overlayImage(clothing_segmented, tshirt_image_no_bg, tshirt_position)

# Make the background white
img_with_white_background = poseDetector.make_background_white(img_with_tshirt)

# Display the original image and the image with overlayed t-shirt side by side
plt.figure(figsize=(10, 5))

# Original image of the man
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Image with overlayed t-shirt and white background
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_with_white_background, cv2.COLOR_BGR2RGB))
plt.title('Image with Overlayed T-Shirt and White Background')
plt.axis('off')

plt.show()
