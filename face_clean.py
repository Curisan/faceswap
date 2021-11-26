"""
    Use face_recognition to clean data, remove non-target face.
    ref: https://github.com/ageitgey/face_recognition
    pros: Simple and easy to use API
    cons: Robustness is poor, unable to recognize non-frontal faces or occluded people
"""

import os
from glob import glob
import shutil

from tqdm import tqdm
import face_recognition

def find_files(directory, pattern):
    return glob(os.path.join(directory, pattern), recursive=True)

def mkdir_ifnot_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

ref_file = "result/face/raw/tao/tao_007921_0.png"
directory = "result/face/raw/tao/"
directory_save = "result/face/clean/tao"
tolerance = 0.5

files = find_files(directory, "*.png")
mkdir_ifnot_exists(directory_save)

ref_image = face_recognition.load_image_file(ref_file)
ref_encoding = face_recognition.face_encodings(ref_image)
if len(ref_encoding)==0:
    print("No face detected, Please use another ref_file.")
    exit(0)
ref_encoding = ref_encoding[0]

for ifile in tqdm(files, desc="Clean face"):
    image = face_recognition.load_image_file(ifile)
    compare_encoding = face_recognition.face_encodings(image)
    if len(compare_encoding)==0:
        continue
    compare_encoding = compare_encoding[0]
    results = face_recognition.compare_faces([ref_encoding], compare_encoding, tolerance=tolerance)
    if results[0]:
        shutil.copy(ifile, directory_save)