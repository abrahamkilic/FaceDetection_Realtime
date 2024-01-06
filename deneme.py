# USAGE
# python detect_faces_emotions.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
from keras.models import load_model
from keras.preprocessing import image

# Argümanları tanımla
DEFAULT_PROTOTXT = "deploy.prototxt.txt"
DEFAULT_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
DEFAULT_CONFIDENCE = 0.5

# Argümanları parse et
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default=DEFAULT_PROTOTXT,
    help=f"path to Caffe 'deploy' prototxt file (default: {DEFAULT_PROTOTXT})")
ap.add_argument("-m", "--model", default=DEFAULT_MODEL,
    help=f"path to Caffe pre-trained model (default: {DEFAULT_MODEL})")
ap.add_argument("-c", "--confidence", type=float, default=DEFAULT_CONFIDENCE,
    help=f"minimum probability to filter weak detections (default: {DEFAULT_CONFIDENCE})")
args = vars(ap.parse_args())

# load face detection model from disk
print("[INFO] loading face detection model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# load emotion detection model from disk
print("[INFO] loading emotion detection model...")
emotion_model = load_model('model_1.h5')

# load the Haar Cascades face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# define label dictionary for emotion detection
label_dict = {0: 'Kizgin', 1: 'İgrenme', 2: 'Korku', 3: 'Mutlu', 4: 'Notr', 5: 'Uzgun', 6: 'Saskin'}

# start the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=900)

    # convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # extract the face region
        face_roi = gray_frame[y:y+h, x:x+w]

        # preprocess the face image for emotion detection
        face_img = cv2.resize(face_roi, (48, 48))
        face_img_array = image.img_to_array(face_img)
        face_img_array = np.expand_dims(face_img_array, axis=0)
        face_img_array
