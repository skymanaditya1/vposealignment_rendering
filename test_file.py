# this is the test file to test everything 
from data.dataset import ImageMeshDataset
from torchvision import transforms
from glob import glob 
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from data.generate_mesh import mesh_from_image
import torch

size = 256

# create the dataloader 
transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

# # have the inverse transformation 
# inverse_transform = transforms.Compose(
#     [
#         transforms.ToPILImage(),
#         transforms.Resize(size),
#         transforms.CenterCrop(size),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0., 0., 0.,], std=[2, 2, 2]),
#         transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1., 1., 1.,])
#     ]
# )

path = '/ssd_scratch/cvit/aditya1/celeba_hq/'

image_paths = glob(path + '/*.jpg')
train_paths, test_paths = train_test_split(image_paths, test_size=0.05)

train_dataset, test_dataset = ImageMeshDataset(train_paths, transform=transform), \
                            ImageMeshDataset(test_paths, transform=transform)

BATCH_SIZE = 4

train_loader = DataLoader(
    train_dataset,
    batch_size = BATCH_SIZE,
    num_workers = 1,
    shuffle = True
)

test_loader = DataLoader(
    test_dataset, 
    batch_size = BATCH_SIZE,
    num_workers = 1
)

batch = next(iter(train_loader))
print(batch.shape)

# the first 3 channels are for the image and the last 3 channels are for the mesh image 

# for the image tensors -- generate the mesh image 
# expected ground truth mesh image -- 3 x 256 x 256

image_tensor = batch[-1][:3]
print(f'original image_tensor range: {torch.min(image_tensor)}, \
    {torch.max(image_tensor)}, {image_tensor.shape}')

mesh_tensor = batch[-1][3:] # this is corresponding to the mesh 
print(f'Mesh tensor properties : {torch.min(mesh_tensor)}, {torch.max(mesh_tensor)}, {mesh_tensor.shape}')
print(f'Unique values original mesh : {torch.unique(mesh_tensor)}')

# convert from range -1 -> 1 to 0 -> 1 
# image_tensor = image_tensor * 0.5 + 0.5
# apply the inverse transformation on the image_tensor before generating the landmarks
# image_tensor = inverse_transform(image_tensor)

# convert from 3 x 256 x 256 -> 256 x 256 x 3
image_tensor = image_tensor.permute(1, 2, 0)

image_array = image_tensor.detach().cpu().numpy()

# normalize the range (0 -> 1) to (0 -> 255)
image_array = (image_array * 255).astype(np.uint8)

print(f'Image from loader : {np.min(image_array[:,:,0])}, {np.max(image_array[:,:,0])}, {np.min(image_array[:,:,1])}, {np.max(image_array[:,:,1])}')

np.savez_compressed('image_saved.npz', data=image_array)

# generate the mesh 
# this expects an image to be given as the input
image_mesh = mesh_from_image(image_array).astype(np.uint8)
# image_mesh = mesh_from_image(image_array)

print(f'mesh properties : {image_mesh.shape}, {np.min(image_mesh)}, {np.max(image_mesh)}, unique : {np.unique(image_mesh)}')

# apply the transformation on the image mesh -- this will take it to the ground truth tensor form
predicted_mesh_tensor = transform(image_mesh)
print(f'Unique values predicted mesh : {torch.unique(predicted_mesh_tensor)}')

print(f'Transform properties : {predicted_mesh_tensor.shape}, {torch.min(predicted_mesh_tensor)}, {torch.max(predicted_mesh_tensor)}')


# code to write the intermediate representations to the disk
to_save_np_array1 = (predicted_mesh_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
to_save_np_array2 = (mesh_tensor.detach().cpu().numpy() * 255).astype(np.uint8)

print(f'original array unique : {np.unique(to_save_np_array1)}')
print(f'predicted array unique : {np.unique(to_save_np_array2)}')

print(f'Saving np arrays to the disk')

np.savez_compressed('predicted_mesh.npz', data=to_save_np_array1)
np.savez_compressed('original_mesh.npz', data=to_save_np_array2)

# compare the shapes of the original and the predicted mesh tensors 
print(f'original tensor : {mesh_tensor.shape}, predicted mesh tensor : {predicted_mesh_tensor.shape}')

# convert the shape of the predicted tensor from 3x256x256 -> 256x256x3
predicted_mesh_tensor = predicted_mesh_tensor.permute(1, 2, 0)

# create the batch of such tensors