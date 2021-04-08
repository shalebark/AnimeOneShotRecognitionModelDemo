from PIL import Image
import dlib
import animeface # you can use other face detectors, only using this one for convenience for the sake of the demo.
from pathlib import Path
import numpy as np

import os, sys

# used to align the face so that it's centered, zoomed so that the left eye is roughly at 55% relative to the image, and the image size is 150x150
from Aligner_Cleaned import Aligner

def get_face_landmarks(image):
    faceboxes = animeface.detect(image)
    assert len(faceboxes) == 1, "Face detector couldn't find one and only one face on the image."

    facebox = faceboxes[0]

    left_eye_center = (int(facebox.left_eye.pos.x + facebox.left_eye.pos.width / 2), int(facebox.left_eye.pos.y + facebox.left_eye.pos.height / 2))
    right_eye_center = (int(facebox.right_eye.pos.x + facebox.right_eye.pos.width / 2), int(facebox.right_eye.pos.y + facebox.right_eye.pos.height / 2))
    eye_center = tuple(np.average([left_eye_center, right_eye_center], axis=0).astype(int))

    return {
        'left-eye-center-pos': left_eye_center,
        'right-eye-center-pos': right_eye_center,
        'eye-center-pos': eye_center,
        'face-box': (facebox.face.pos.x, facebox.face.pos.y, facebox.face.pos.width, facebox.face.pos.height)
    }

aligner = Aligner()
model_path = str(Path(__file__).resolve().parent / 'models/facial_portrait_only_3_29_21.dat')
encoder = dlib.face_recognition_model_v1(model_path)

def extract_face(image):
    landmarks = get_face_landmarks(image)
    aligned_image = aligner.align_and_extract_face(np.asarray(image), landmarks)
    return aligned_image

def compare_images(image1, image2, threshold=0.6):
    encode1 = np.array(encoder.compute_face_descriptor(extract_face(image1), 1))
    encode2 = np.array(encoder.compute_face_descriptor(extract_face(image2), 1))
    return np.linalg.norm([encode1] - encode2, axis=1)[0] <= 0.6

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Aligns Facial Features of Animated Characters')
    parser.add_argument('image1', metavar='Image', type=Path, help='First image.')
    parser.add_argument('image2', metavar='Image', type=Path, help='Second image to compare to image1.')

    args = parser.parse_args()

    # convert RGBA images to RGB
    image1 = Image.open(args.image1).convert('RGB')
    image2 = Image.open(args.image2).convert('RGB')

    print(compare_images(image1, image2))
