import tkinter as tk
import tkinter.messagebox
import customtkinter
import cv2

import backend

def create_ui():
    # Create the Tkinter window
    root = tk.Tk()
    root.geometry("1000x1100")
    root.title("AR Clothing")

    # Load the image
    image_path = "Assest/Intro/intro_img.png"
    root.image = tk.PhotoImage(file=image_path)

    # Function to stop camera capture and close the application
    def close_application():
        # Stop camera capture
        # Add code here to stop the camera capture, if applicable
        # For example:
        # camera.release()

        # Close the Tkinter window
        root.destroy()

    # Bind the close button (X) to the close_application function
    root.protocol("WM_DELETE_WINDOW", close_application)

    def initialize_ui():
        # Display the image on the left side
        image_label = tk.Label(root, image=root.image)
        image_label.grid(row=0, column=0, padx=20, pady=20)

        # Create a CTkFrame on the right side for details input
        details_frame = customtkinter.CTkFrame(root, width=300, height=300, corner_radius=10)
        details_frame.grid(row=0, column=1, padx=20, pady=20)

        # Add labels and entry widgets for details input
        height_label = tk.Label(details_frame, text="Height:")
        height_label.grid(row=0, column=0, padx=10, pady=10)

        height_entry = customtkinter.CTkEntry(details_frame, width=50, corner_radius=5)
        height_entry.grid(row=0, column=1, padx=10, pady=10)

        weight_label = tk.Label(details_frame, text="Weight:")
        weight_label.grid(row=1, column=0, padx=10, pady=10)

        weight_entry = customtkinter.CTkEntry(details_frame, width=50, corner_radius=5)
        weight_entry.grid(row=1, column=1, padx=10, pady=10)

        age_label = tk.Label(details_frame, text="Age:")
        age_label.grid(row=2, column=0, padx=10, pady=10)

        age_entry = customtkinter.CTkEntry(details_frame, width=50, corner_radius=5)
        age_entry.grid(row=2, column=1, padx=10, pady=10)

        def button_function():
            if not all([height_entry.get(), weight_entry.get(), age_entry.get()]):
                tk.messagebox.showerror("Error", "Please fill all the details.")
            else:
                # Fetch details and call backend function
                wt = weight_entry.get()
                ht = height_entry.get()
                age = age_entry.get()
                backend.run_backend(wt, ht, age)

        button = customtkinter.CTkButton(master=root, text="Click here to start Camera", corner_radius=10,
                                         command=button_function, width=250)
        button.grid(row=3, column=1)

    # Run the Tkinter event loop
    initialize_ui()
    # Instead of root.mainloop(), use root.mainloop() at the end to ensure the function returns the UI HTML
    root.mainloop()

    # Since Tkinter's mainloop() doesn't return, the function will continue here only after the window is closed
    # You can return any HTML content here if needed
    return "<h1>UI created successfully</h1>"
