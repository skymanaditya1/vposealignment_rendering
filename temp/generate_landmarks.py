# code to generate the landmarks
import torch
import face_alignment 

import skimage.io as io

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

path = '/ssd_scratch/cvit/aditya1/celeba_hq/000004.jpg'

image = io.imread(path)

landmarks = fa.get_landmarks_from_image(image)


print(len(landmarks))

# generate landmarks from torch tensor 
image_tensor = torch.tensor(image)

landmarks = fa.get_landmarks_from_image(image_tensor)

# estimate the landmarks of the face
