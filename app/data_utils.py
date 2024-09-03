# utils/dicom_utils.py
import pydicom
from PIL import Image
import numpy as np
import torch


def load_dicom_image(dicom_file_path: str) -> np.ndarray:
    dicom = pydicom.dcmread(dicom_file_path)
    input_img = dicom.pixel_array

     # normalize image data
    input_img[input_img < 0] = 0
    epsilon = 1e-10
    min_val = np.min(input_img)
    max_val = np.max(input_img)
    input_img = (input_img - min_val) / (max_val - min_val + epsilon)

    return np.expand_dims((np.expand_dims(input_img,0)),0)

def save_png(image_array: np.ndarray, output_path: str):

    image = Image.fromarray(image_array)
    image.save(output_path)
