import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing import Optional
from .loss_funcs import BarlowTwinsLoss


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 10, capacity: int = 64) -> None:
        super().__init__()

        # Convolutional section
        self.conv1_ = nn.Conv2d(
            in_channels=1,
            out_channels=capacity,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.conv2_ = nn.Conv2d(
            in_channels=capacity,
            out_channels=capacity * 2,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.fc_ = nn.Linear(
            in_features=capacity * 2 * 7 * 7,
            out_features=latent_dim
        )
    
    def forward(self, x):
        x = F.relu(self.conv1_(x))
        x = F.relu(self.conv2_(x))
        # Flatten batch of multi-channel feature maps to a batch of feature vectors
        x = x.view(x.size(0), -1)
        x = self.fc_(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 10, capacity: int = 64) -> None:
        super().__init__()
        self.capacity_ = capacity
        self.fc_ = nn.Linear(
            in_features=latent_dim,
            out_features=capacity * 2 * 7 * 7
        )
        self.conv2_ = nn.ConvTranspose2d(
            in_channels=capacity * 2,
            out_channels=capacity,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.conv1_ = nn.ConvTranspose2d(
            in_channels=capacity,
            out_channels=1,
            kernel_size=4,
            stride=2,
            padding=1
        )
    
    def forward(self, x):
        x = self.fc_(x)
        # Unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = x.view(x.size(0), self.capacity_ * 2, 7, 7)
        x = F.relu(self.conv2_(x))
        # Last layer before output is tanh, since the images are normalized and 0-centered
        x = torch.tanh(self.conv1_(x))
        return x

class AutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc_ = Encoder()
        self.dec_ = Decoder()
    
    def forward(self, x):
        latent = self.enc_(x)
        recon = self.dec_(latent)
        return recon


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, batch_size: int, loss_fn: str = "mse") -> None:
        super().__init__()
        self.encoder_ = Encoder()
        self.decoder_ = Decoder()
        if loss_fn.lower() == "barlow":
            self.loss_fn_ = BarlowTwinsLoss(batch_size)
        else:
            self.loss_fn_ = nn.MSELoss()

        # Save model's hyperparameters passed to init
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, _ = batch
        z = self.encoder_(x)
        x_hat = self.decoder_(z)
        loss = self.loss_fn_(x_hat, x)
        self.log("loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        x, _ = batch
        z = self.encoder_(x)
        x_hat = self.decoder_(z)
        val_loss = self.loss_fn_(x_hat, x)
        self.log("val_loss", val_loss)
    
    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        x, _ = batch
        z = self.encoder_(x)
        x_hat = self.decoder_(z)
        test_loss = self.loss_fn_(x_hat, x)
        self.log("test_loss", test_loss)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer