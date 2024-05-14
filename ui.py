import cv2
import numpy as np
import backend

# Global variable to track button click
button_clicked = False

def on_mouse(event, x, y, flags, params):
    global button_clicked
    btn_x, btn_y, btn_width, btn_height = params
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click is within the button region
        if btn_x <= x <= btn_x + btn_width and btn_y <= y <= btn_y + btn_height:
            button_clicked = True
    elif event == cv2.EVENT_MOUSEMOVE:
        # Check if mouse is within the button region
        if btn_x <= x <= btn_x + btn_width and btn_y <= y <= btn_y + btn_height:
            cv2.setMouseCallback("AR Clothing", on_mouse_hover, params)
        else:
            cv2.setMouseCallback("AR Clothing", on_mouse, params)

def on_mouse_hover(event, x, y, flags, params):
    global button_clicked
    btn_x, btn_y, btn_width, btn_height = params
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click is within the button region
        if btn_x <= x <= btn_x + btn_width and btn_y <= y <= btn_y + btn_height:
            button_clicked = True
    elif event == cv2.EVENT_MOUSEMOVE:
        # Check if mouse is outside the button region
        if not (btn_x <= x <= btn_x + btn_width and btn_y <= y <= btn_y + btn_height):
            cv2.setMouseCallback("AR Clothing", on_mouse, params)

def display_ui(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Resize the image to decrease window size
    scale_percent = 40  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # Text and button dimensions
    text = "Click here to Try outfits"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    btn_padding = 10  # Padding around the text in the button
    btn_width = text_size[0] + 2 * btn_padding  # Button width adjusted to text size
    btn_height = text_size[1] + 2 * btn_padding  # Button height adjusted to text size
    btn_x, btn_y = int(image.shape[1] * 7/9.5) - btn_width // 2, int(image.shape[0] * 0.85)  # Positioned on the right side

    # Draw button with gray background
    btn_color = (113, 121, 126)  # Gray color
    cv2.rectangle(image, (btn_x, btn_y), (btn_x + btn_width, btn_y + btn_height), btn_color, -1)

    # Display the image
    cv2.putText(image, text, (btn_x + btn_padding, btn_y + btn_height - btn_padding), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.imshow("AR Clothing", image)

    # Set initial mouse callback function
    cv2.setMouseCallback("AR Clothing", on_mouse, (btn_x, btn_y, btn_width, btn_height))

    while True:
        if button_clicked:
            cv2.destroyAllWindows()
            backend.run_backend()  # Call backend code function
            return True
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    return False