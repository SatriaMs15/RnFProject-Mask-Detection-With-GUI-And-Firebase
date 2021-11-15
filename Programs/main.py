from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import firebase_config
import MLFunction
import Graph
import UploadFunction
import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import pyrebase
import datetime

# creating main application window
root = tk.Tk()
root.title("Face Mask Detection")

#  Frame for GUI
top_frame = Frame(root, bd=10)
top_frame.pack()

middle_frame = Frame(root, bd=10)
middle_frame.pack()

bottom_frame = Frame(root, bd=10)
bottom_frame.pack()

# Initalized firebase
config = firebase_config.fb_config()
firebase = pyrebase.initialize_app(config)
database = firebase.database()

# define a video capture object
vid = cv2.VideoCapture(0)

# Load model and cascade
model, cascade = MLFunction.load_model()


# Define function to show frame
def show_frames():
    # Get the latest frame and convert into Image
    cv2image = cv2.cvtColor(vid.read()[1], cv2.COLOR_BGR2RGB)
    MLFunction.face_detection(cv2image, model, cascade)
    img = Image.fromarray(cv2image)

    # Convert image to PhotoImage
    imgtk = ImageTk.PhotoImage(image=img)

    label.imgtk = imgtk
    label.configure(image=imgtk)

    # Repeat after an interval to capture continiously
    label.after(20, show_frames)


def graph():
    # Create a Graph
    fig = Graph.addGraph()
    # creating the Tkinter canvas

    # containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig, master=top_frame)
    canvas.draw()

    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().grid(row=1, column=1, padx=20)


# upload image
def upload():
    gray_upload = cv2.cvtColor(vid.read()[1], cv2.COLOR_BGR2GRAY)
    faces_upload = cascade.detectMultiScale(gray_upload, 1.01, 3, minSize=[120, 120])
    for x, y, w, h in faces_upload:
        # Get Detectet Face Location
        face_upload = vid.read()[1][y:y + h, x:x + w]
        encode = cv2.imencode('.jpg', face_upload)[1]
        UploadFunction.encode_upload(database, encode)
        break


# Get Recent Data
def get_img():
    # Get Recent Data Image
    Key_image = []
    data_img = database.child("Img").get()
    for item in data_img.each():
        img_get = item.key()
        Key_image.append(str(img_get))

    # Show Image only if image in the database have 3 image or more
    if len(Key_image) >= 3:

        # Sort to the recent image
        Key_image = sorted(Key_image, key=lambda date: datetime.datetime.strptime(date, "%m-%d-%Y %H:%M:%S"), reverse=True)
        Key_image = Key_image[0:3]
        # Prepare 3 Data to show it in GUI
        data_img1 = database.child("Img").child(Key_image[0]).get()
        data_img1 = data_img1.val()["img_detect"]
        data_img1 = UploadFunction.decode_img(data_img1)

        data_img2 = database.child("Img").child(Key_image[1]).get()
        data_img2 = data_img2.val()["img_detect"]
        data_img2 = UploadFunction.decode_img(data_img2)

        data_img3 = database.child("Img").child(Key_image[2]).get()
        data_img3 = data_img3.val()["img_detect"]
        data_img3 = UploadFunction.decode_img(data_img3)

        # Update variabel Recent Image 1
        img1.configure(image=data_img1)
        img1.image = data_img1  # keep a reference!

        # Update variabel Recent Image 2
        img2.configure(image=data_img2)
        img2.image = data_img2  # keep a reference!

        # Update variabel Recent Image 3
        img3.configure(image=data_img3)
        img3.image = data_img3  # keep a reference!


# Title Face Detection
title_fc = Label(top_frame, text="Face Detection")
title_fc.grid(row=0, column=0)

# Title Counting People
title_cp = Label(top_frame, text="Counting People")
title_cp.grid(row=0, column=1)

# Create a Label to capture the Video frames
label = Label(top_frame)
label.grid(row=1, column=0)

# Run Graph
graph()

# Run video show
show_frames()

# Code For button upload
btn_img_upload = Button(top_frame, text='Upload', command=upload, bg="white", fg="black")
btn_img_upload.grid(row=2, column=0, pady=10)

# Code For button graph_update
btn_Update_Graph = Button(top_frame, text='Update_Graph', command=graph, bg="white", fg="black")
btn_Update_Graph.grid(row=2, column=1, pady=10)

# Code For Update Image
btn_img_get = Button(middle_frame, text='Get Recent Image', command=get_img, bg="white", fg="black")
btn_img_get.grid(row=0, column=3)

# Recent Image 1
img1 = Label(middle_frame)
img1.grid(row=1, column=1, padx=20, pady=20)

# Recent Image 2
img2 = Label(middle_frame)
img2.grid(row=1, column=3, padx=20, pady=20)

# Recent Image 3
img3 = Label(middle_frame)
img3.grid(row=1, column=5, padx=20, pady=20)

get_img()

# mainloop dari trinket
root.mainloop()
