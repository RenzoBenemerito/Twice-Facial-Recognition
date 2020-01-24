"""
clean.py
Authored by: Renzo Benemerito

A Script for cleaning image data for facial recognition.
Deletes corrupted images and GIFs but saves the first frame of GIFs.
The dataset should be contained in a directory called data.
Each class directory should be in the data directory.
"""

# Import the necessary packages
import os
from PIL import Image

img_dir = r"data"

for classes in os.listdir(img_dir):
    for file in os.listdir("data/"+classes):
        # Check for GIFs
        if file.endswith(".gif") or file.endswith(".GIF"):
            new_filename = img_dir + "/" + classes + "/" + file[:-4] + ".jpg"
            Image.open(img_dir + "/" + classes + "/" + file).convert('RGB').save(new_filename)
            os.remove(img_dir + "/" + classes + "/" + file)
            print("[DELETED] {}".format(file))
            file = file[:-4] + ".jpg"
        # Check for corrupted files
        try :
            with Image.open(img_dir + "/" + classes + "/" + file) as im:
                print('Ok {}',format(file))
        except :
            print("[DELETED] {}".format(file))
            os.remove(img_dir + "/" + classes + "/" + file)