import ui
import backend

if __name__ == "__main__":
    image_path = "Assest/Intro/intro_img.png"
    # text = "Welcome To\nAR Clothing\nClick here to start"
    if ui.display_ui(image_path):
        # Run the backend code here
        backend.run_backend()
