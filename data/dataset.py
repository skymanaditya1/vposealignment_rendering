# Custom Dataset class
from torch.utils.data import Dataset
import random

import cv2
import numpy as np
import torch
from data.generate_mesh import landmark_from_image, drawPolylines

from data.image_utils import resize_frame, write_np_file

# initialize the FAN based face alignment network
# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device=device, flip_input=False)

class ImageMeshDataset(Dataset):
    def __init__(self, paths, fa, transform=None):
        self.transform = transform
        self.paths = paths
        self.fa = fa
    
    def __getitem__(self, idx):
        image_path = self.paths[idx]
        image = resize_frame(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), 256)
        
        # generate a random pose 
        pose_image_path = self.paths[random.randint(0, len(self.paths)-1)]

        # mesh_image = mesh_from_image(image)
        # image_landmarks = landmark_from_image(image, self.fa)
        landmarks = landmark_from_image(image, self.fa)

        # in case landmarks were not detected from the image
        if landmarks is None or len(landmarks) == 0:
        # if len(landmarks) == 0:
            return self.__getitem__(random.randint(0, self.__len__()-1))

        landmarks = landmarks[0] # take only the first landmarks of the detected landmarks

        # generate the conditional mesh from the landmarks
        mesh_image = np.ones(image.shape, dtype=np.uint8) * 255
        drawPolylines(mesh_image, landmarks)

        # if mesh_image is None:
        #     return self.__getitem__(random.randint(0, self.__len__()-1))

        # mesh_image = mesh_image.astype(np.uint8)

        # np.savez_compressed('inside_dataset_mesh.npz', data=mesh_image)

        # print(f'Inside dataset : {np.min(mesh_image)}, {np.max(mesh_image)}, {np.unique(mesh_image)}')

        # convert to torch tensor and concatenate the image and target mesh representation
        # concatenated = np.concatenate((image, mesh_image), axis=2)

        if self.transform:
            image = self.transform(image)
            # mesh_image = self.transform(mesh_image)
            # landmarks = self.transform(np.array(landmarks, dtype=np.uint8))
            landmarks = torch.tensor(landmarks)
            # print(f'Landmarks inside the dataset : {landmarks.shape}')
            mesh_image = self.transform(mesh_image)

            # # convert the range from 0 -> 1 to -1 -> 1
            # image = (image - 0.5) * 2
            # mesh_image = (mesh_image - 0.5) * 2

            # print(f'Unique properties image : {torch.unique(image)}')
            # print(f'Unique properties mesh : {torch.unique(mesh_image)}')

        assert image.shape == mesh_image.shape

        concatenated = torch.cat([image, mesh_image], axis=0)
        # print(f'Concatenated shape : {concatenated.shape}')

        return concatenated, landmarks


    def __len__(self):
        return len(self.paths)