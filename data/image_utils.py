# file with the image utility functions 
import cv2
import numpy as np

def resize_frame(frame, resize_dim=256):
    return cv2.resize(frame, (resize_dim, resize_dim), interpolation=cv2.INTER_AREA)

def write_np_file(filename, np_data):
    np.savez_compressed(filename, data=np_data)