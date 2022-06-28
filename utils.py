from data.dataset import ImageMeshDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from glob import glob

from loss import VQLPIPS

import face_alignment

def get_image_mesh_loader(path, fa, transform, device):
    from models.vqvae import VQVAE

    image_paths = glob(path + '/*.jpg')
    train_paths, test_paths = train_test_split(image_paths, test_size=0.05)

    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device=device, flip_input=False)

    train_dataset, test_dataset = ImageMeshDataset(train_paths, fa, transform=transform), \
                                ImageMeshDataset(test_paths, fa, transform)

    BATCH_SIZE = 32

    # set multiprocessing context here 
    # multiprocessing_context='spawn'

    train_loader = DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        num_workers = 0,
        shuffle = True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size = BATCH_SIZE,
        num_workers = 0
    )

    # the image and mesh are concatenated along the channel dimension
    model = VQVAE(3*2).to(device)
    lpips = VQLPIPS().to(device)

    return train_loader, test_loader, model, lpips

def get_loaders_and_models(type, path, fa, transform, device):
    if type == 1:
        return get_image_mesh_loader(path, fa, transform, device)