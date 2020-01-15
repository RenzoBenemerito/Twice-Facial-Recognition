import os
from PIL import Image

img_dir = r"data"

for classes in os.listdir(img_dir):
    for file in os.listdir("data/"+classes):
        if file.endswith(".gif") or file.endswith(".GIF"):
            new_filename = img_dir + "/" + classes + "/" + file[:-4] + ".jpg"
            Image.open(img_dir + "/" + classes + "/" + file).convert('RGB').save(new_filename)
            os.remove(img_dir + "/" + classes + "/" + file)
            file = file[:-4] + ".jpg"
            print(file)
      
        try :
            with Image.open(img_dir + "/" + classes + "/" + file) as im:
                print('ok')
        except :
            print(img_dir + "/" + classes + "/" + file)
            os.remove(img_dir + "/" + classes + "/" + file)