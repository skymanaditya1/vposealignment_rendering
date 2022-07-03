import argparse
import sys
import os

import torch
from torch import nn, optim

from torchvision import transforms, utils

from tqdm import tqdm

from scheduler import CycleScheduler
import distributed as dist

from utils import get_loaders_and_models

import numpy as np

from torch.multiprocessing import set_start_method

size = 256

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

# inverse transformation applied on the tensors
inverse_transform = transforms.Compose(
    [
        transforms.Normalize(mean=[0., 0., 0.,], std=[2, 2, 2]),
        transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1., 1., 1.,])
    ]
)
        
def train(epoch, loader, model, lpips_model, optimizer, scheduler, device):
    loader = tqdm(loader)

    # criterion = nn.MSELoss(reduction='none')
    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    recon_loss_weight = 1
    perceptual_loss_weight = 0.25
    sample_size = 25

    for i, (image_mesh, target_image) in enumerate(loader):
        model.zero_grad()

        image_mesh = image_mesh.to(device)
        target_image = target_image.to(device)

        out, latent_loss = model(image_mesh)
        latent_loss = latent_loss.mean()

        predicted_image = out[:,:3]

        # compute the reconstruction loss between target_image and predicted_image 
        recon_loss = criterion(predicted_image, target_image)

        # compute the perceptual loss between the predicted image and target image 
        perceptual_loss = lpips_model(predicted_image, target_image)

        loss = recon_loss_weight * recon_loss \
                + latent_loss_weight * latent_loss \
                + perceptual_loss_weight * perceptual_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr = optimizer.param_groups[0]["lr"]

        loader.set_description(
            (
                f"epoch : {epoch+1}; recon_loss : {recon_loss.item():.5f}; "
                f"latent : {latent_loss.item():.3f}; perceptual loss : {perceptual_loss.item():.5f} "
                f"lr : {lr:.5f}"
            )
        )

        validate_at = 100
        # this is the validation component
        if i % validate_at == 0:
            print(f'Inside validation, generating validation examples')
            model.eval()

            # sample = img[:sample_size]
            sample = image_mesh[:sample_size] # dims - sample_size x 6 x 256 x 256
            target_image = target_image[:sample_size]

            with torch.no_grad():
                out, _ = model(sample)

            # out will have the dimension -> sample_size x 6 x 256 x 256
            predicted = out[:, :3] # first 3 values gives the predicted image with the pose modified
            sample_image = sample[:, :3] # first 3 channels contains the image information
            target_mesh = sample[:, 3:]

            # most probably the normalization scheme is wrong 
            sample_size = min(sample_size, len(sample_image))

            utils.save_image(
                torch.cat([sample_image, predicted, target_mesh, target_image], 0),
                f"sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                nrow=sample_size,
                normalize=True,
                range=(-1, 1),
            )

            model.train()

# requires the dataloader, model, and validation_dir where results are stored
def validation(test_loader, model, validation_dir, device, sample_images=25):
    # perform the predictions using the test_loader
    model.eval()

    with torch.no_grad():
        mesh_images, target_images = next(iter(test_loader))

        # print(mesh_images.shape, target_images.shape)
        
        sample_images = min(sample_images, len(mesh_images))

        mesh_images = mesh_images[:sample_images]
        target_images = target_images[:sample_images]

        mesh_images = mesh_images.to(device)
        target_images = target_images.to(device)

        out, _ = model(mesh_images)

        # print(f'Out shape : {out.shape}')

        predictions = out[:, :3]
        source_images = mesh_images[:, :3]
        target_meshes = mesh_images[:, 3:]

        # save the source image, target mesh, predicted image, and the target image
        for i in range(len(source_images)):
            source_image = source_images[i]
            target_mesh = target_meshes[i]
            predicted_image = predictions[i]
            target_image = target_images[i]

            # print(source_image.shape, target_mesh.shape, predicted_image.shape, target_image.shape)

            # save the images to the disk 
            utils.save_image(
                torch.cat([source_image, target_mesh, predicted_image, target_image], 1),
                f"{validation_dir}/sample_{str(i).zfill(3)}.png",
                nrow=1, # indicates each item goes into its own row
                normalize=True,
                range=(-1, 1),
            )

def main(args):
    device = "cuda"

    path = args.path
    train_loader, test_loader, model, lpips_model = get_loaders_and_models(1, path, transform, device)

    model = model.to(device)
    lpips_model = lpips_model.to(device)

    if args.test:
        # perform validation and store the results
        # load the model given the checkpoint
        checkpoint_path = args.ckpt
        model.load_state_dict(torch.load(checkpoint_path))

        print(f'Pretrained model {checkpoint_path} loaded successfully')

        NUM_SAMPLES = 25
        os.makedirs(args.sample_dir)
        print(f'Saving results at : {args.sample_dir}')
        validation(test_loader, model, args.sample_dir, device, NUM_SAMPLES)

    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = None
        if args.sched == "cycle":
            scheduler = CycleScheduler(
                optimizer,
                args.lr,
                n_iter=len(train_loader) * args.epoch,
                momentum=None,
                warmup_proportion=0.05,
            )

        for i in range(args.epoch):
            train(i, train_loader, model, lpips_model, optimizer, scheduler, device)

            torch.save(model.state_dict(), f"checkpoint/vqvae_{str(i + 1).zfill(3)}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--sample_dir", type=str, default=None)
    parser.add_argument("path", type=str)

    args = parser.parse_args()

    def get_random(num_chars=5):
        import random
        chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
        return ''.join([chars[random.randint(0, len(chars)-1)] for i in range(num_chars)])

    if args.sample_dir is None:
        args.sample_dir = '/ssd_scratch/cvit/aditya1/pose_alignment_results/validation_samples_' + get_random()        

    print(args)

    main(args)