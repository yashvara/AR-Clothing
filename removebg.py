import cv2
import numpy as np
from rembg import remove

# Load the man image
man_img = cv2.imread("Assest/Reference/man2.png")

# Load the t-shirt image with the background removed
shirt_img = cv2.imread("Assest/Clothes/tshirt1.png")
gray = cv2.cvtColor(shirt_img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros_like(gray)
cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
result = cv2.bitwise_and(shirt_img, shirt_img, mask=mask)

# Resize the t-shirt image to fit the man image
result_resized = cv2.resize(result, (man_img.shape[1], man_img.shape[0]))

# Invert the binary mask for the t-shirt to create a mask for the background
bg_mask = cv2.bitwise_not(mask)

# Ensure that the binary mask has the same size as the man image
bg_mask = cv2.resize(bg_mask, (man_img.shape[1], man_img.shape[0]))

# Convert the background mask to 3 channels to match the man image
bg_mask = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2BGR)

# cv2.imshow('Overlay Result', bg_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Bitwise AND operation to extract the background from the man image
bg_removed_man = cv2.bitwise_and(man_img, bg_mask)

# cv2.imshow('Overlay Result', bg_removed_man)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Bitwise OR operation to overlay the t-shirt on the man image
output_img = cv2.bitwise_or(bg_removed_man, result_resized)

out = remove(shirt_img)

# Display the result
# cv2.imshow('Original Man Image', man_img)
cv2.imshow('rembg Result', out)
cv2.waitKey(0)
cv2.destroyAllWindows()

