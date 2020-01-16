"""
Facial Recognition Encode Script

"""

# import the necessary packages
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import dlib
import cv2
import face_recognition
import pickle
import os
import argparse

# Initialize arguments parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '--filename',
    '-f',
    type=str,
    help="Path to embeddings pickle file.",
    default="data.pickle")
args = parser.parse_args()

# Load the encodings if they already exist and load the embeddings and labels to lists
# Create a black pickle file for writing if not
print("[INFO] loading encodings...")
try:
    data = pickle.loads(open(args.filename, "rb").read())
    known_encodings = data["encodings"]
    known_names = data["names"]

except Exception as e:
    print("[INFO] Error", e)
    data = open(args.filename, "wb")
    known_encodings = []
    known_names = []

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shapepredictor/shape_predictor_68_face_landmarks.dat')

# Function for creating embedings for our images
# append the embeddings and labels to our respective lists
# write the embeddings and labels to a pickle file
def encode(crop, name):
    global data
    data_to_encode = {}
    try:
        print("[INFO] encoding face...")
        encoding = face_recognition.face_encodings(crop)
        known_encodings.append(encoding[0])
        name = name
        known_names.append(name)
        data_to_encode = {"encodings": known_encodings, "names": known_names}
        with open(args.filename, "wb") as f:
            f.write(pickle.dumps(data_to_encode))
        return "Successfully Encoded the image"
    except Exception as e:
        return e

# Function for face detection
# This calls our encode function on each detected face to generate embeddings
def detect():
    x=y=h=w=0
    for dirs in os.walk("data/", topdown=True):
        for d in dirs[1]:
            if not os.path.exists("data_processed/"+d):
                os.mkdir("data_processed/"+d)
            for files in os.walk(dirs[0]+d+"/", topdown=True):
                for f in files[2]:
                    path = files[0]+f
                    frame = cv2.imread(path)
                    print(path)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    rects = detector(gray, 0)
                    colored = frame.copy()
                    height, width = colored.shape[:2]

                    
                    for rect in rects:
                        # get the coordinates of the face and enlarge the bounding box
                        x = rect.left()-25
                        y = rect.top()-50
                        w = rect.right()+25
                        h = rect.bottom()+25
                        
                        if y < 0:
                            y = 0

                        if x < 0:
                            x = 0
                        
                        if y > height:
                            y = height
                        
                        if x > width:
                            x = width
                        
                        if h > height:
                            h = height

                        if w > width:
                            w = width
                        
                        cv2.rectangle(frame, (x, y), (w,h), (0, 255, 0), 2)
                        roi_gray = gray[y:h, x:w]
                        roi_colored = colored[y:h, x:w]
                        status = encode(roi_colored, d)
                        cv2.imwrite("data_processed/"+d+"/"+f,roi_colored)
                        print(status)
                    cv2.imshow("Frame", roi_colored)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
	
    

if __name__ == '__main__':
    if not os.path.exists("data_processed"):
        os.mkdir("data_processed")
    detect()
	