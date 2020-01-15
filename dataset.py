from google_images_download import google_images_download
import os
import argparse

arg = argparse.ArgumentParser()
arg.add_argument("-k", "--keyword", required=True,
	help="keyword you want to search")
arg.add_argument("-n", "--number", required=True,
	help="number of images you want to download")
arg.add_argument("-o", "--output_dir", default=None,
    help="output directory where images should be saved")

args = arg.parse_args()

if not os.path.exists("data/"):
    os.mkdir("data/")

response = google_images_download.googleimagesdownload()   #class instantiation

keyword = args.keyword
limit = args.number
output_dir = args.output_dir
if output_dir is None:
    output_dir = keyword

arguments = {"keywords":keyword,"limit":limit,"format":"jpg","output_directory":"data/","image_directory":output_dir,"print_urls":True}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images