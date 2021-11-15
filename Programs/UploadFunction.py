from tkinter import *
import cv2
import numpy as np
from PIL import Image, ImageTk
from base64 import b64decode, b64encode
import datetime


# Encode Image and find counting and upload it to firebase
def encode_upload(database, encode):
    e = datetime.datetime.now()
    # Img_Upload
    coded_image = b64encode(encode).decode()
    data_img = {"img_detect": coded_image}
    database.child("Img").child(e.strftime("%m-%d-%Y %H:%M:%S")).set(data_img)

    # Update Counting

    # Check if there is already Today Date Date, If not we making new, if exist we update it
    data_exist = False

    data_Date = database.child("Counting").get()

    # If thare is a Data
    for item in data_Date.each():
        date_get = str(item.key())
        if date_get == e.strftime("%m-%d-%Y"):
            date = database.child("Counting").child(date_get).get()
            num_count = date.val()["count"]
            num_count += 1

            date_update = {"count": num_count}
            database.child("Counting").child(date_get).update(date_update)
            data_exist = True
            break

    # If not
    if data_exist != True:
        date_new = {"count": 1}
        database.child("Counting").child(e.strftime("%m-%d-%Y")).set(date_new)


# Decode Image from Firebase
def decode_img(path):
    decoded_image = b64decode(str(path))
    jpg_as_np = np.frombuffer(decoded_image, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)
    img = cv2.resize(img, (160, 160))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    return img
