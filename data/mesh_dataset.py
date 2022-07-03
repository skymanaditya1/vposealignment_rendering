'''
In the mesh_dataset.py file, we are sampling an image from a directory,
and another image from the same directory would be used as the ground-truth 
the target mesh would be computed from the ground-truth image
'''
from torch.utils.data import Dataset

import random
import cv2
import numpy as np
import torch
from glob import glob

from data.generate_mesh import landmark_from_image, drawPolylines, mesh_from_image
from data.image_utils import resize_frame, write_np_file

class MeshDataset(Dataset):
    def __init__(self, dir_paths, transform=None):
        self.transform = transform

        # this gives the dir paths 
        self.dir_paths = dir_paths
    
    def __getitem__(self, idx):
        dir_path = self.dir_paths[idx]

        # get the images from the dir path 
        images = glob(dir_path + '/*.jpg')

        if len(images) == 0:
            return self.__getitem__(random.randint(0, self.__len__()-1))
        
        # load two different indices from the images list 
        source_idx, target_idx = torch.randperm(len(images))[:2]
        source_idx, target_idx = source_idx.item(), target_idx.item()

        source_image_path, target_image_path = images[source_idx], images[target_idx]

        source_image = resize_frame(cv2.cvtColor(cv2.imread(source_image_path), cv2.COLOR_BGR2RGB), 256)
        target_image = resize_frame(cv2.cvtColor(cv2.imread(target_image_path), cv2.COLOR_BGR2RGB), 256)
        
        # generate the mesh from the target image
        target_mesh = mesh_from_image(target_image)

        if target_mesh is None:
            return self.__getitem__(random.randint(0, self.__len__()-1))

        target_mesh = target_mesh.astype(np.uint8)

        # convert to torch tensor and concatenate the image and target mesh representation 
        if self.transform:
            image_tensor = self.transform(source_image)
            mesh_tensor = self.transform(target_mesh)

            concatenated = torch.cat([image_tensor, mesh_tensor], dim=0) # concatenates channel wise        

            target_tensor = self.transform(target_image)

        return concatenated, target_tensor


    def __len__(self):
        return len(self.dir_paths)