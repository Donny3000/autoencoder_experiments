#!/usr/bin/env python3

import os
import torch
import torch.nn.functional as F
import torch.utils.data as data
import pytorch_lightning as pl

from argparse import ArgumentParser
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from numpy.random import RandomState
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA, NMF
from skimage.metrics import structural_similarity as ssim

from models import cnn_ae

root_chkpt_dir = Path("chkpts")


if __name__ == "__main__":
    parser = ArgumentParser(description="Autoencoder Experiments for Learning")
    parser.add_argument("-b", "--batch-size", type=int,  default=256,      help="The batch size to use for training.")
    parser.add_argument("-c", "--chkpt-name", type=str,  default="cnn_ae", help="Name of the folder within the chkpts directory to contain the logging information for the training session")
    parser.add_argument("-l", "--loss-func",  type=str,  choices=["mse", "barlow"], default="mse", help="Which loss function to use for the model.")
    parser.add_argument("-m", "--model",      type=Path,                   help="Path to a model checkpoint or pretrained model.")
    args = vars(parser.parse_args())

    print(args)

    # Increment checkpoint directory, if necessary
    num_chkpt_dirs = len(
        list(
            root_chkpt_dir.rglob(f"*args['chkpt_name']*")
        )
    )
    chkpt_dir_name = f"{args['chkpt_name']}_{num_chkpt_dirs:02d}"
        

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

    autoencoder = cnn_ae.LitAutoEncoder(
        batch_size=args["batch_size"],
        loss_fn=args["loss_func"]
    )
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=-1,
        default_root_dir='chkpts',
        callbacks=[
            ModelCheckpoint(
                dirpath=root_chkpt_dir / chkpt_dir_name,
                filename='{epoch}-{loss:.5f}-{val_loss:.5f}'
            ),
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
