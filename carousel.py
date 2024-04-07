import cv2
import numpy as np
import os

def display_carousel(images, box_size=100):
    # Calculate the number of rows and columns for the grid layout
    num_rows = len(images) // 3
    if len(images) % 3 != 0:
        num_rows += 1

    # Create a black canvas for the carousel
    carousel = np.zeros((num_rows * box_size, 300, 3), dtype=np.uint8)

    # Paste each image onto the carousel in a grid format
    idx = 0
    for i in range(num_rows):
        for j in range(3):
            if idx < len(images):
                img = images[idx]
                carousel[i * box_size:(i + 1) * box_size, j * box_size:(j + 1) * box_size] = img
                idx += 1

    # Display the carousel window
    cv2.imshow("Carousel", carousel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Define the path to the clothes directory
ClothsPath = "Assest/Clothes"

# List all files in the directory
clothesList = os.listdir(ClothsPath)
print(clothesList)

# Load each image from the directory
images = []
for file_name in clothesList:
    file_path = os.path.join(ClothsPath, file_name)
    img = cv2.imread(file_path)
    images.append(img)

# Display the carousel of images
display_carousel(images, box_size=100)
