from rembg import remove
from PIL import Image
import cv2

input_path = 'Assest/Clothes/tshirt1.png'

# Read the input image using OpenCV
input_image = cv2.imread(input_path)

# Display the input image
input_pil_image = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
input_pil_image.show()

# Use rembg to remove the background
output_image = remove(input_image)

# Convert the output array to a PIL Image
output_pil_image = Image.fromarray(output_image)

# Display the output image
output_pil_image.show()
