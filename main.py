#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from argparse import ArgumentParser
from pathlib import Path
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from numpy.random import RandomState
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA, NMF
from skimage.metrics import structural_similarity as ssim

from models import cnn_ae

plt.ion()
image_shape = 64, 64
rng = RandomState(0)

def load_face_data():
    """Download and return Olivetti face dataset with the data globally/locally centered

    Returns:
        _type_: _description_
    """
    faces, _ = fetch_olivetti_faces(return_X_y=True, shuffle=True, random_state=rng)
    n_samples, n_features = faces.shape

    # Global centering (focus on one feature, centering all samples)
    faces_centered = faces - faces.mean(axis=0)

    # Local centering (focus on one sample, centering all features)
    faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)

    return faces, faces_centered, n_samples, n_features


def plot_gallery(title, images, n_col=3, n_row=2, cmap=plt.cm.gray):
    fig, axs = plt.subplots(
        nrows=n_row,
        ncols=n_col,
        figsize=(2.0 * n_col, 2.3 * n_row),
        facecolor="white",
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02, hspace=0, wspace=0)
    fig.set_edgecolor("black")
    fig.suptitle(title, size=16)
    for ax, vec in zip(axs.flat, images):
        vmax = max(vec.max(), -vec.min())
        im = ax.imshow(
            vec.reshape(image_shape),
            cmap=cmap,
            interpolation="nearest",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.axis("off")

    fig.colorbar(im, ax=axs, orientation="horizontal", shrink=0.99, aspect=40, pad=0.01)
    plt.show()

def train_epoch(model: nn.Module, device, dataloader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer):
    model.train()
    train_loss = []

    for img_batch, _ in dataloader:
        img_batch = img_batch.to(device)
        recon = model(img_batch)
        loss = loss_fn(recon, img_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())
    
    return np.mean(train_loss)


def test_epoch(model: nn.Module, device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    model.eval()

    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []

        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)

            # Encode data
            recon_data = model(image_batch)

            # Append the network output and the original image to the lists
            conc_out.append(recon_data.cpu())
            conc_label.append(image_batch.cpu())

        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)

        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)

    return val_loss.data


def plot_ae_outputs(fig, model, n=10):
    # plt.figure(figsize=(16,4.5))
    targets = test_dataset.targets.numpy()
    t_idx = {i:np.where(targets==i)[0][0] for i in range(n)}

    for i in range(n):
      ax = plt.subplot(2,n,i+1)
      img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
      model.eval()

      with torch.no_grad():
         rec_img  = model(img)

      plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)

      if i == n//2:
        ax.set_title('Original images')
    
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  

      if i == n//2:
         ax.set_title('Reconstructed images')

    plt.show()

def show_image(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


if __name__ == "__main__":
    parser = ArgumentParser(description="Autoencoder Experiments for Learning")
    parser.add_argument("-b", "--batch-size", type=int,  default=256, help="The batch size to use for training.")
    parser.add_argument("-m", "--model",      type=Path,              help="Path to a model checkpoint or pretrained model.")
    args = vars(parser.parse_args())

    # faces, faces_centered, n_samples, n_features = load_face_data()
    # plot_gallery("Faces from dataset", faces_centered[:(image_shape[0] * image_shape[1])])

    transform = transforms.ToTensor()
    train_set = MNIST(root="MNIST", download=True, train=True, transform=transform)
    train_set_size = int(len(train_set) * 0.8)
    valid_set_size = len(train_set) - train_set_size

    # Split the train set into two
    seed = torch.manual_seed(0)
    train_set, valid_set = data.random_split(
        train_set,
        [train_set_size, valid_set_size],
        generator=seed
    )
    test_set = MNIST(root="MNIST", download=True, train=False, transform=transform)
    train_dataloader = DataLoader(
        dataset=train_set,
        num_workers=os.cpu_count(),
        batch_size=args["batch_size"]
    )
    valid_dataloader = DataLoader(
        dataset=valid_set,
        num_workers=os.cpu_count(),
        batch_size=args["batch_size"]
    )

    autoencoder = cnn_ae.LitAutoEncoder()
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=-1,
        default_root_dir='chkpts',
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min")
        ]
    )
    trainer.fit(
        model=autoencoder,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
        # Restore the model, epoch, step, LR schedules, apex, etc...
        ckpt_path=args["model"].as_posix() if args["model"] is not None else None
    )
    trainer.test(
        model=autoencoder,
        dataloaders=DataLoader(test_set)
    )

    ### Define the standard loss function
    # loss_fn = nn.MSELoss()

    # ### Set the random seed for reproducible results
    # torch.manual_seed(0)

    # cnnae = cnn_ae.AutoEncoder()

    # optim = torch.optim.Adam(cnnae.parameters(), lr=1e-3, weight_decay=1e-5)

    # # Check if the GPU is available
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # print(f"Selected Device: {device}")

    # ### Move both encoder and decoder to the selected device
    # cnnae.to(device)

    # num_params = sum(
    #     p.numel() for p in cnnae.parameters() if p.requires_grad
    # )
    # print(f"Number of parameters: {num_params}")

    # plt.ion()
    # fig = plt.figure(figsize=(16,4.5))

    # num_epochs = 30
    # diz_loss = {'train_loss': [], 'val_loss': []}
    # for epoch in range(num_epochs):
    #     train_loss = train_epoch(cnnae, device, train_loader, loss_fn, optim)
    #     val_loss = test_epoch(cnnae, device, test_loader, loss_fn)
    #     print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs, train_loss, val_loss))
    #     diz_loss['train_loss'].append(train_loss)
    #     diz_loss['val_loss'].append(val_loss)
    #     plot_ae_outputs(fig, cnnae, n=10)
    
    # test_epoch(cnnae, device, test_loader, loss_fn).item()
    
    # # Plot losses
    # plt.figure(figsize=(10,8))
    # plt.semilogy(diz_loss['train_loss'], label='Train')
    # plt.semilogy(diz_loss['val_loss'], label='Valid')
    # plt.xlabel('Epoch')
    # plt.ylabel('Average Loss')

    # #plt.grid()
    # plt.legend()

    # #plt.title('loss')
    # plt.show()
