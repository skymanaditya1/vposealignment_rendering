from data.mesh_dataset import MeshDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from glob import glob

from loss import VQLPIPS


def get_image_mesh_loader(path, transform, device):
    from models.vqvae import VQVAE

    # dir_paths = glob(path + '/*')
    dir_paths = glob(path + '/*/*/*/[0-9][0-9][0-9][0-9][0-9]')
    train_paths, test_paths = train_test_split(dir_paths, test_size=0.05)

    train_dataset, test_dataset = MeshDataset(train_paths, transform), MeshDataset(test_paths, transform)

    BATCH_SIZE = 16

    train_loader = DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        num_workers = 2,
        shuffle = True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size = BATCH_SIZE,
        num_workers = 1
    )

    # the image and mesh are concatenated along the channel dimension
    model = VQVAE(3*2).to(device)
    lpips = VQLPIPS().to(device)

    return train_loader, test_loader, model, lpips

def get_loaders_and_models(type, path, transform, device):
    if type == 1:
        return get_image_mesh_loader(path, transform, device)