import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained DeepLab model
model = tf.keras.models.load_model('deeplab_model.h5')

# Load your image
img = cv2.imread('Assest/Reference/man2.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, (128, 128))  # Resize according to the model's input size
img_normalized = img_resized / 255.0  # Normalize to [0, 1]

# Predict the segmentation mask
mask = model.predict(np.expand_dims(img_normalized, axis=0))

# The output is a 4D tensor, we remove the first dimension using np.squeeze
mask = np.squeeze(mask)

# The output of the model is probabilities for each class. We take the class with the highest probability for each pixel
mask = np.argmax(mask, axis=-1)

# Now, `mask` is a 2D array with the same height and width as the input image,
# and each pixel has a value corresponding to the class it belongs to.
