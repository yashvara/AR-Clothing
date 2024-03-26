import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'Assest/Reference/man2.png'
image = cv2.imread(image_path)
if image is not None:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_clothes_color = np.array([0, 40, 40])
    upper_clothes_color = np.array([30, 255, 255])
    clothes_mask = cv2.inRange(hsv_image, lower_clothes_color, upper_clothes_color)
    clothes_only = cv2.bitwise_and(image, image, mask=clothes_mask)

    # cv2.imshow('Original Image', image)
    # cv2.imshow('Segmented Clothes', clothes_only)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    plt.imshow(cv2.cvtColor(clothes_only, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Turn off axis
    plt.show()

else:
    print(f"Error: Unable to read the image at '{image_path}'")

# man_img = cv2.imread("Assest/Reference/man2.png")
# shirt_img = cv2.imread("Assest/Clothes/tshirt1.png")
