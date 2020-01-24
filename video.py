"""
video.py
Authored by: Renzo Benemerito

A script inspired by this blog post: https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/
Takes a video as input
Processes each frame and generates bounding boxes and labels
The processed frames are written to an output video
"""

# import the necessary packages
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import numpy as np
import seaborn as sns

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", type=str,
	help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
ap.add_argument("-m", "--model", type=str, default="twice-model.pickle",
	help="path to svm model", required=True)
ap.add_argument("-t", "--threshold", type=float, default=0.5,
	help="threshold for recognition")
args = vars(ap.parse_args())

# initialize the pointer to the video file and the video writer
print("[INFO] processing video...")
stream = cv2.VideoCapture(args["input"])
fps = stream.get(cv2.CAP_PROP_FPS)
print("[INFO] processing at {} fps".format(fps))
writer = None

with open(args["model"], 'rb') as infile:
    (model, names) = pickle.load(infile)

# Function for predicting a face cropped by our HOG face detector
def svm(model, crop):
    name = "Unknown"
    try:
        proba = model.predict_proba([crop]).ravel()
        threshold = max(proba)
        id_proba = 9
		# if the probability is greater than our threshold, accept the prediction
        if threshold > args["threshold"]:
            id_proba = np.argmax(proba)
            name = model.classes_[id_proba]
    except Exception as e:
        print("[INFO] There was an error", e)
    return (name, id_proba)


# Generate unique colors for each label
colors = []
palette = sns.color_palette("hls", len(model.classes_)+1)
palette = np.array(palette) * 255

# loop over frames from the video file stream
while True:
	# grab the next frame
	(grabbed, frame) = stream.read()

	# if the frame was not grabbed, then we have reached the
	# end of the stream
	if not grabbed:
		break

	# convert the input frame from BGR to RGB then resize it to have
	# a width of 750px (to speedup processing)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(frame, width=750)
	r = frame.shape[1] / float(rgb.shape[1])

	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input frame, then compute
	# the facial embeddings for each face
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []
	id_probas = []
	# loop over the facial embeddings
	for encoding in encodings:
		name, id_proba = svm(model, encoding)
		
		# update the list of names
		names.append(name)
		id_probas.append(id_proba)
	# loop over the recognized faces
	for ((top, right, bottom, left), name, id_proba) in zip(boxes, names, id_probas):
		# rescale the face coordinates
		top = int(top * r)
		right = int(right * r)
		bottom = int(bottom * r)
		left = int(left * r)
		# draw the predicted face name on the image
		if name != "Unknown":
			cv2.rectangle(frame, (left, top), (right, bottom),
				palette[id_proba], 2)
			y = top - 15 if top - 15 > 15 else top + 15
			cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
				0.75, palette[id_proba], 2)

	# if the video writer is None *AND* we are supposed to write
	# the output video to disk initialize the writer
	if writer is None and args["output"] is not None:
		fourcc = cv2.VideoWriter_fourcc(*"mp4v")
		writer = cv2.VideoWriter(args["output"], fourcc, fps,
			(frame.shape[1], frame.shape[0]), True)

	# if the writer is not None, write the frame with recognized
	# faces t odisk
	if writer is not None:
		writer.write(frame)

	# check to see if we are supposed to display the output frame to
	# the screen
	if args["display"] > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

# close the video file pointers
stream.release()

# check to see if the video writer point needs to be released
if writer is not None:
	writer.release()
