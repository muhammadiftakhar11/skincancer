import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import cv2
import numpy as np
import threading
import os

# ---------------- APP SETUP ---------------- #

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Skin Cancer Detection System")
app.geometry("900x720")
app.resizable(True, True)

# ---------------- LOAD MODEL ---------------- #

model_path = "skin_model.h5"

if not os.path.exists(model_path):
    raise FileNotFoundError("Model file not found!")

model = tf.keras.models.load_model(model_path)

# ---------------- FUNCTIONS ---------------- #

def predict_image(file_path):

    try:

        # Show uploaded image
        img = Image.open(file_path)
        img.thumbnail((350,350))
        img_tk = ImageTk.PhotoImage(img)

        image_label.configure(image=img_tk)
        image_label.image = img_tk

        progress_bar.set(0)

        # Loading animation
        for i in range(101):
            progress_bar.set(i/100)
            app.update_idletasks()

        # Prepare image
        img_cv = cv2.imread(file_path)
        img_cv = cv2.resize(img_cv,(224,224))
        img_cv = img_cv / 255.0
        img_cv = np.reshape(img_cv,(1,224,224,3))

        predictions = model.predict(img_cv)[0]

        # Fix error (single output model)
        cancer_prob = float(predictions[0]) * 100

        # Risk classification
        if cancer_prob <= 40:
            level = "NORMAL"
            color = "green"

        elif cancer_prob <= 70:
            level = "MEDIUM RISK"
            color = "black"

        else:
            level = "HIGH RISK"
            color = "red"

        result_text = f"""
Cancer Level : {level}
Probability  : {cancer_prob:.2f} %
"""

        result_label.configure(text=result_text,text_color=color)

    except Exception as e:
        messagebox.showerror("Error",str(e))


def upload_image():

    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files","*.jpg *.png *.jpeg")]
    )

    if file_path:
        threading.Thread(target=predict_image,args=(file_path,)).start()


def reset_ui():

    image_label.configure(image="")
    image_label.image = None

    result_label.configure(
        text="Analysis result will appear here",
        text_color="black"
    )

    progress_bar.set(0)

# ---------------- MAIN FRAME ---------------- #

main_frame = ctk.CTkFrame(app,fg_color="transparent")
main_frame.pack(expand=True)

# ---------------- TITLE ---------------- #

title = ctk.CTkLabel(
    main_frame,
    text="Skin Cancer Detection System",
    font=("Arial",32,"bold")
)

title.pack(pady=40)

# ---------------- UPLOAD BUTTON ---------------- #

upload_btn = ctk.CTkButton(
    main_frame,
    text="Upload Skin Image",
    width=260,
    height=50,
    command=upload_image
)

upload_btn.pack(pady=15)

# ---------------- IMAGE PREVIEW ---------------- #

image_label = ctk.CTkLabel(main_frame,text="")
image_label.pack(pady=20)

# ---------------- RESULT TEXT ---------------- #

result_label = ctk.CTkLabel(
    main_frame,
    text="Analysis result will appear here",
    font=("Arial",20),
    justify="center"
)

result_label.pack(pady=15)

# ---------------- PROGRESS BAR ---------------- #

progress_bar = ctk.CTkProgressBar(
    main_frame,
    width=420,
    height=18
)

progress_bar.set(0)
progress_bar.pack(pady=20)

# ---------------- RESET BUTTON ---------------- #

reset_btn = ctk.CTkButton(
    main_frame,
    text="Reset",
    width=200,
    height=45,
    fg_color="orange",
    hover_color="#ff914d",
    command=reset_ui
)

reset_btn.pack(pady=25)

# ---------------- RUN APP ---------------- #

app.mainloop()