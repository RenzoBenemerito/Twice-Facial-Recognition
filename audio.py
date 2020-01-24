"""
audio.py
Authored by: Renzo Benemerito

Dependencies: ffmpeg
Make sure ffmpeg.exe is in this directory

This script adds audio to our processed video
"""

# Import required packages
import subprocess
import argparse
import os

# Initialize arguments parser
parser = argparse.ArgumentParser()

parser.add_argument(
    '--input',
    '-i',
    type=str,
    help="Path input video.",
    required=True
    )

parser.add_argument(
    '--audio',
    '-a',
    type=str,
    help="File name of extracted audio from video.",
    required=True
    )

parser.add_argument(
    '--processed',
    '-p',
    type=str,
    help="Path processed video.",
    required=True
    )

parser.add_argument(
    '--output',
    '-o',
    type=str,
    help="Path to output video.",
    required=True
    )

args = parser.parse_args()

# Initialize variables
input_video = args.input
processed_video = args.processed
extracted_audio = args.audio
output_video = args.output

# Call ffmpeg
extract = "ffmpeg -i {} {}".format(input_video,extracted_audio)
merge = "ffmpeg -i {} -i {} -c:v copy -vcodec copy {}".format(extracted_audio,processed_video,output_video)
subprocess.call(extract,shell=True)
subprocess.call(merge,shell=True)
os.remove(processed_video)