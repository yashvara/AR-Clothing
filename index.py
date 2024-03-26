import tkinter as tk
from tkinter import filedialog, Toplevel, messagebox
from PIL import ImageTk, Image
import staticimg2

class ClothingTryOnApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Clothing Try-On App")
        self.master.geometry("500x400")  # Set initial window size

        # Load model image
        self.model_image_path = "C:/Users/dell/PycharmProjects/ARClothing/Assest/Reference/man2.png"
        self.model_image = Image.open(self.model_image_path)
        self.model_image = self.model_image.resize((300, 400), Image.LANCZOS)
        self.model_photo = ImageTk.PhotoImage(self.model_image)

        # Model image frame
        self.model_frame = tk.Frame(self.master)
        self.model_frame.grid(row=0, column=0, padx=10, pady=10)
        self.model_label = tk.Label(self.model_frame, image=self.model_photo)
        self.model_label.pack()

        # Clothing frame
        self.clothing_frame = tk.Frame(self.master)
        self.clothing_frame.grid(row=0, column=1, padx=10, pady=10)

        # Choose clothing button
        self.choose_clothing_button = tk.Button(self.clothing_frame, text="Choose Clothing", command=self.choose_clothing)
        self.choose_clothing_button.pack(pady=10)

        # Chosen clothing image label
        self.chosen_clothing_label = tk.Label(self.clothing_frame)
        self.chosen_clothing_label.pack()

    def choose_clothing(self):
        # Open file dialog to choose clothing image
        clothing_path = filedialog.askopenfilename(initialdir="/", title="Select Clothing Image",
                                                   filetypes=(("Image Files", "*.jpg *.png *.jpeg"), ("All Files", "*.*")))
        if clothing_path:
            try:
                # Call the add_tshirt_to_image() function from staticimg2
                img_with_tshirt = staticimg2.add_tshirt_to_image(clothing_path)

                # Display the image in a new window
                self.display_image_in_window(img_with_tshirt)
            except Exception as e:
                print("Error : ", {e})
                messagebox.showerror("Error", f"Error occurred: {e}")

    def display_image_in_window(self, img):
        # Convert the numpy array to a PIL Image
        img_pil = Image.fromarray(img)

        # Create a new window
        window = Toplevel(self.master)
        window.title("Image with T-Shirt")

        # Resize image if necessary to fit in window
        if img_pil.width > 800 or img_pil.height > 400:
            img_pil.thumbnail((800, 400))

        # Display the image in the new window
        img_tk = ImageTk.PhotoImage(img_pil)
        label = tk.Label(window, image=img_tk)
        label.image = img_tk  # Keep a reference to prevent garbage collection
        label.pack()

def main():
    root = tk.Tk()
    app = ClothingTryOnApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
