import cv2
import backend

# Global variable to track button click
button_clicked = False

def on_mouse_click(event, x, y, flags, params):
    global button_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click is within the button region
        btn_x, btn_y, btn_width, btn_height = params
        if btn_x <= x <= btn_x + btn_width and btn_y <= y <= btn_y + btn_height:
            button_clicked = True

def display_ui(image_path, text):
    # Read the image
    image = cv2.imread(image_path)

    # Resize the image to decrease window size
    scale_percent = 40  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # Display welcome text
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (int(image.shape[1]/2) + 50, int(image.shape[0]/2))
    fontScale = 1
    color = (255, 255, 255)
    thickness = 2
    image = cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA)

    # Add "Try On" button
    btn_width, btn_height = 75, 25
    btn_x, btn_y = int(image.shape[1]/2) - btn_width // 2, int(image.shape[0] * 0.8)
    cv2.rectangle(image, (btn_x, btn_y), (btn_x + btn_width, btn_y + btn_height), (0, 0, 255), -1)
    cv2.putText(image, "Try On", (btn_x + 5, btn_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the image
    cv2.imshow("AR Clothing", image)

    # Set mouse callback function
    cv2.setMouseCallback("AR Clothing", on_mouse_click, (btn_x, btn_y, btn_width, btn_height))

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
