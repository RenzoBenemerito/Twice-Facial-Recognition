import cv2
import pickle
import dlib
import face_recognition
import argparse
from sklearn import svm
import os

parser = argparse.ArgumentParser()

parser.add_argument(
    '--input',
    '-i',
    type=str,
    help="Path to embeddings pickle file.",
    required=True
    )

parser.add_argument(
    '--output',
    '-o',
    type=str,
    help="Path to SVM model.",
    required=True
    )

args = parser.parse_args()

def train_svm():
    file_name = args.input
    model_name = args.output
    try:
        data = pickle.loads(open(file_name, "rb").read())
        encodings = data["encodings"]
        names = data["names"]

    except Exception as e:
        print("[INFO] Error", e)
        data = open(file_name, "wb")
        encodings = []
        names = []
        exit()
    # Create and train the SVC classifier
    clf = svm.SVC(gamma="scale", C=1.0, kernel="linear", probability=True)
    clf.fit(encodings,names)
    model = clf
    print("[INFO] Labels", names)
    print("[INFO] Successfully Trained the Network.")
    with open(model_name, 'wb') as outfile:
        pickle.dump((clf, names), outfile)

try:
    train_svm()
except Exception as e:
    print("[ERROR] Error encountered. Provide more classes.", e)