import argparse
import sys
import os

import torch
from torch import nn, optim
from data.generate_mesh import landmark_from_batch

from torchvision import transforms, utils

from tqdm import tqdm

from scheduler import CycleScheduler
import distributed as dist

from utils import get_loaders_and_models

import face_alignment

import numpy as np

from torch.multiprocessing import set_start_method

size = 256

# transform = transforms.Compose(
#         [
#             transforms.ToPILImage(),
#             transforms.Resize(size),
#             transforms.CenterCrop(size),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
#         ]
#     )

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
        
def train(epoch, loader, model, lpips_model, optimizer, scheduler, fa, device):
    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.MSELoss(reduction='none')

    latent_loss_weight = 0.25
    recon_loss_weight = 1
    perceptual_loss_weight = 0.25
    sample_size = 25

    # mse_sum = 0
    # mse_n = 0

    perceptual_losses = list()
    recon_losses = list()

    for i, (image_mesh, landmarks) in enumerate(loader):
        model.zero_grad()

        image_mesh = image_mesh.to(device)

        out, latent_loss = model(image_mesh)

        # reconstruction of the landmarks and similarity in perceptual quality
        prediction = out[:, :3] # batch_size x 3 x img_dim x img_dim

        input_image = image_mesh[:, :3]
        # input_mesh = image_mesh[:, 3:]

        # predicted_mesh = generate_mesh(prediction).to(device) # generates the mesh for the predicted image
        # generate the landmarks of the predicted image 
        # apply inverse transform on the predicted image 
        inverse_prediction = inverse_transform(prediction)*255

        # generate the landmarks on the predictions
        predicted_landmarks, mask = landmark_from_batch(inverse_prediction, fa) # dimension -> batch_size x 68 x 3

        predicted_landmarks_tensor = torch.tensor(predicted_landmarks)

        # for computing the landmark loss, loss between just the landmarks can be computed

        # print(f'Shapes, predicted landmarks : {predicted_landmarks_tensor.shape}, input landmarks : {landmarks.shape}')

        # assert predicted_landmarks_tensor.shape == input_mesh.shape

        # compute the landmark loss between the input landmarks and the predicted landmarks
        recon_mesh_loss = criterion(predicted_landmarks_tensor, landmarks)

        # apply the mask on the recon loss and reduce by mean
        # check if the mask is completely True in which case loss is set to 0
        # recon_loss = recon_mesh_loss[mask].abs().mean()
        
        loss_zero = True
        for i in range(len(mask)):
            if mask[i] is False:
                loss_zero = False
                break

        if loss_zero:
            recon_loss = criterion(torch.tensor(1.), torch.tensor(1.)).abs().mean()
        else:
            # compute the loss by first applying the mask
            recon_loss = recon_mesh_loss[mask].abs().mean()

        # print(f'The loss is set to : {recon_loss}')

        # recon_mesh_loss = criterion(predicted_mesh, input_mesh)

        # measure the perceptual / identity loss between input_image and out
        # TODO - Add identity loss
        perceptual_loss = lpips_model(prediction, input_image)

        latent_loss = latent_loss.mean()

        loss = recon_loss_weight * recon_loss + \
            perceptual_loss_weight * perceptual_loss + \
                latent_loss_weight * latent_loss

        # loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        # compute the part losses 
        # part_perceptual_sum = perceptual_loss.item() * input_image.shape[0]
        # part_recon_sum = recon_mesh_loss.item() * input_image.shape[0]

        # part_mse_n = input_image.shape[0] # current batch size

        perceptual_losses.append(perceptual_loss.item())
        recon_losses.append(recon_loss.item())

        # part_mse_sum = recon_loss.item() * img.shape[0]
        # part_mse_n = img.shape[0]
        # comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        # comm = dist.all_gather(comm)

        # for part in comm:
        #     mse_sum += part["mse_sum"]
        #     mse_n += part["mse_n"]

        # aggregate the stats on the primary node
        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"epoch : {epoch+1}; recon_loss : {recon_loss.item():.5f}; "
                    f"latent : {latent_loss.item():.3f}; avg recon : {np.average(np.array(recon_losses)):.5f}; "
                    f"avg perceptual : {np.average(np.array(perceptual_losses)):.3f}; "
                    f"lr : {lr:.5f}"
                )
            )

            # loader.set_description(
            #     (
            #         f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
            #         f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
            #         f"lr: {lr:.5f}"
            #     )
            # )

            # this is the validation component
            if i % 100 == 0:
                model.eval()

                # sample = img[:sample_size]
                sample = image_mesh[:sample_size] # dims - sample_size x 6 x 256 x 256

                with torch.no_grad():
                    out, _ = model(sample)

                # out will have the dimension -> sample_size x 6 x 256 x 256
                predicted = out[:,:3] # first 3 values gives the predicted image with the pose modified
                sample_image = sample[:, :3] # first 3 channels contains the image information

                utils.save_image(
                    torch.cat([sample_image, predicted], 0),
                    f"sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                    nrow=sample_size,
                    normalize=True,
                    range=(-1, 1),
                )

                model.train()


def main(args, fa):
    device = "cuda"

    args.distributed = dist.get_world_size() > 1

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    # dataset = datasets.ImageFolder(args.path, transform=transform)
    # sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
    # loader = DataLoader(
    #     dataset, batch_size=128 // args.n_gpu, sampler=sampler, num_workers=2
    # )

    path = args.path
    train_loader, test_loader, model, lpips_model = get_loaders_and_models(1, path, fa, transform, device)

    # model = VQVAE().to(device)
    # define face_alignment

    # This creates multiple replicas of the model
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

        lpips_model = nn.parallel.DistributedDataParallel(
            lpips_model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )


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
        train(i, train_loader, model, lpips_model, optimizer, scheduler, fa, device)

        if dist.is_primary():
            torch.save(model.state_dict(), f"checkpoint/vqvae_{str(i + 1).zfill(3)}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

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
    parser.add_argument("path", type=str)

    args = parser.parse_args()

    print(args)

    # initialize the face alignment library here
    # initialize the fa depending on the instance
    fas = [face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device="cuda:{}".format(id)) for id in range(args.n_gpu)]
    
    # print the local rank 
    print(f'The local rank is : {dist.get_local_rank()}')

    # face alignment object corresponding to rank
    print(f'Initializing the face alignment repository')
    fa = fas[dist.get_local_rank()]

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,fa))
