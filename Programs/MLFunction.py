import tensorflow as tf
import cv2
import numpy as np


# Load Model and haar detection
def load_model():
    # Load model machine learning
    model = tf.keras.models.load_model('ModelMDMobileNet.h5')

    # Load Haar detection Classifier
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    return model, cascade


# Run Face Detection
def face_detection(frame, model, cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Play here if you want custom the detection quality
    # second parameter -> put in range 1.1 below Exampel : 1.01,1.05,1.005,etc
    #                  -> much smaller second parameter means mor good prediction but slower camera movement
    # third parameter (neighbor) -> put postive Int value :1,2,3,.... . it will check if there is a neighbour
    # fourth parameter (minSize) -> put matrix, it is for limit minimum size of face that count as face
    #                            -> more size means face that in long distance at camera would be not detected
    faces = cascade.detectMultiScale(gray, 1.01, 3, minSize=[150, 150])

    if faces is not None:
        for x, y, w, h in faces:
            # Get Detectet Face Location
            face_image = frame[y:y + h, x:x + w]
            # Preprocess
            resize_img = cv2.resize(face_image, (320, 320))
            normalized = resize_img / 255.0
            reshape = np.reshape(normalized, (1, 320, 320, 3))
            reshape = np.vstack([reshape])
            # predict
            result = model.predict(reshape)

            # Make Bounding Box
            if result[0][0] > result[0][1]:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.rectangle(frame, (x, y - 50), (x + w, y), (0, 255, 0), -1)
                cv2.putText(frame, "With Mask", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.rectangle(frame, (x, y - 50), (x + w, y), (0, 0, 255), -1)
                cv2.putText(frame, "Without Mask", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
