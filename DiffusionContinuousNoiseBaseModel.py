import math
from abc import ABC, abstractmethod

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
import math


class DiffusionBaseModelContinuousNoise(pl.LightningModule):
    """Architecture of the base diffusion model.

    In this version, we add continuous noise to the input data.
    Continous in this case refers to the fact that the noise is not added according to the
    discrete time steps, but rather according to a continuous or real valued time step.

    The code has been implemented by following the official implementation of the paper:
    "On the Importance of Noise Scheduling for Diffusion Models" by Ting Chen (2023)

    Attributes:
        input_size: (int) size of the input data. Note height and width are the same
        reverse_steps: (int) number of steps to be used in the reverse pass
        conditional: (bool) whether the model is conditional or not.
            The conditioning mechanism is described in the follow-up paper:
            Saharia et al. "Image super-resolution via iterative refinement."
            arXiv preprint arXiv:2103.16774 (2021).
        loss_type: (str) type of loss function. Can be one of mse, l1
    """

    def __init__(
        self,
        input_size: int = 500,
        reverse_steps: int = 2000,
        conditional: bool = True,
        loss_type: str = "mse",
        normalize: bool = True,
        noise_schedule: str = "cosine",  # can be linear, cosine or sigmoid
    ):
        """Inits DiffusionBaseModel with the parameters from the config file.

        Args:
            input_size: (int) size of the input data. Note height and width are the same
            reverse_steps: (int) number of steps to be used in the reverse sampling from the trained model
            input_channels: (int) number of channels in the input data
            output_channels: (int) number of channels in the output data
            conditional: (bool) whether the model is conditional or not.
                The conditioning mechanism used here is described in the paper:
                Saharia et al. "Image super-resolution via iterative refinement."
            loss_type: (str) type of loss function. Can be one of mse, l1
            normalize: (bool) whether to normalize the input data or not. Defaults to True.
                The input is normalized to have a variance of 1 as it improved the performance.



        """
        super().__init__()

        self.input_size = input_size
        self.reverse_steps = reverse_steps
        self.conditional = conditional
        self.loss_type = loss_type
        self.normalize = normalize
        self.noise_schedule = noise_schedule

    @abstractmethod
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, x_condition=None
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: (torch.Tensor) input data with noise added according to timestep
            t: (torch.Tensor) Tensor of diffusion time steps scaled to [0, 1]s
            x_condition: (torch.Tensor, optional) conditioning data. Defaults to None.

        Returns:
            (torch.Tensor) output of the forward pass

        """
        raise NotImplementedError(
            "Every model needs to have a forward pass. Please Implement this function"
        )

    def linear_gamma_schedule(self, t, clip_min=1e-9):
        """Linear schedule for the diffusion coefficient gamma.

        Unlike the original paper, in this schedule, the gamma makes use of
        real valued time steps. This is done by interpolating between the
        discrete time steps.

        Args:
            t: (torch.Tensor) Tensor of diffusion time steps scaled to [0, 1]s
            clip_min: (float, optional) minimum value of the diffusion coefficient. Defaults to 1e-9.

        Returns:
            (torch.Tensor) diffusion coefficient
        """
        gamma = 1 - t
        return torch.clip(gamma, clip_min, 1.0)

    def cosine_gamma_schedule(self, t, start=0, end=1, tau=1, clip_min=1e-9):
        """Cosine schedule for the diffusion coefficient gamma.

        Unlike the original paper, in this schedule, the gamma makes use of
        real valued time steps. This is done by interpolating between the
        discrete time steps.

        Args:
            t: (torch.Tensor) Tensor of diffusion time steps scaled to [0, 1]s
            start: (float) start of the cosine function
            end: (float) end of the cosine function
            tau: (float) steepness of the cosine function
            clip_min: (float, optional) minimum value of the diffusion coefficient. Defaults to 1e-9.

        Returns:
            (torch.Tensor) diffusion coefficient
        """
        v_start = math.cos(start * math.pi / 2) ** (2 * tau)
        v_end = math.cos(end * math.pi / 2) ** (2 * tau)
        gamma = torch.cos((t * (end - start) + start) * torch.pi / 2) ** (2 * tau)
        gamma = (v_end - gamma) / (v_end - v_start)
        return torch.clip(gamma, clip_min, 1.0)

    def sigmoid_gamma_schedule(self, t, start=-3, end=3, tau=1.0, clip_min=1e-9):
        # A gamma function based on sigmoid function.
        v_start = sigmoid(start / tau)
        v_end = sigmoid(end / tau)
        gamma = torch.sigmoid((t * (end - start) + start) / tau)
        gamma = (v_end - gamma) / (v_end - v_start)
        return torch.clip(gamma, clip_min, 1.0)

    def get_gamma_schedule(self, t, schedule="linear", **kwargs):
        """Returns the diffusion coefficient gamma.

        Args:
            t: (torch.Tensor) Tensor of diffusion time steps scaled to [0, 1]s
            schedule: (str) type of schedule to be used. Can be one of linear, sigmoid or cosine
            kwargs: (dict) additional arguments for the schedule

        Returns:
            (torch.Tensor) diffusion coefficient
        """
        if schedule == "linear":
            return self.linear_gamma_schedule(t, **kwargs)
        elif schedule == "sigmoid":
            return self.sigmoid_gamma_schedule(t, **kwargs)
        elif schedule == "cosine":
            return self.cosine_gamma_schedule(t, **kwargs)
        else:
            raise ValueError(f"Unknown schedule type: {schedule}")

    def pad_timesteps_to_match_batch(self, x, t):
        """Reshape timesteps from ([batch_size] -> [batch_size, 1, 1,1])
        Args:
            x: (torch.Tensor) tensor to be padded
            t: (torch.Tensor) tensor to match
        Returns:
            (torch.Tensor) padded tensor
        """
        padding_dims = x.ndim - t.ndim
        if padding_dims <= 0:
            return t
        return t.view(*t.shape, *((1,) * padding_dims))

    def get_loss(self, batch, condition_batch=None, scale=1, normalize=True):
        """Calculates the loss function.

        This is based on Algorithm 2 in the paper:
        "The Importance of Noise Scheduling for Diffusion Models" by Ting Chen (2023)

        Args:
            batch: (torch.Tensor) input data. In our case, it is the noised sharp images.
            condition_batch: (torch.Tensor, optional) conditioning data. Defaults to None.

        Returns:
            (torch.Tensor) loss value
        """

        timesteps = torch.rand(
            size=[batch.shape[0]], device=self.device
        )  # shape = [batch_size]
        timesteps_padded = self.pad_timesteps_to_match_batch(
            batch, timesteps
        )  # shape = [batch_size, 1, 1, 1]

        epsilon = torch.randn(batch.shape, device=self.device)

        noise_level = self.get_gamma_schedule(
            timesteps_padded, schedule=self.noise_schedule
        )
        noised_image = (
            torch.sqrt(noise_level) * scale * batch
            + torch.sqrt(1 - noise_level) * epsilon
        )

        if normalize:
            noised_image = noised_image / torch.std(
                noised_image, dim=[1, 2, 3], keepdim=True
            )

        if condition_batch is not None:
            epsilon_predicted = self.forward(
                noised_image, timesteps.unsqueeze(-1).type(torch.float), condition_batch
            )
        else:
            epsilon_predicted = self.forward(
                noised_image, timesteps.unsqueeze(-1).type(torch.float)
            )

        if self.loss_type == "l1":
            loss = nn.functional.l1_loss(
                epsilon_predicted.reshape(-1, self.input_size),
                epsilon.reshape(-1, self.input_size),
            )
        elif self.loss_type == "mse":
            loss = nn.functional.mse_loss(
                epsilon_predicted.reshape(-1, self.input_size),
                epsilon.reshape(-1, self.input_size),
            )
        else:
            raise ValueError("Invalid loss type")
        return loss

    def ddpm_step(self, x_t, eps_pred, t_now, t_next):
        """DDPM step as described in the paper:
        "Diffusion Probabilistic Models" by Ho et al. (2021)

        Args:
            x_t: (torch.Tensor) current image
            eps_pred: (torch.Tensor) predicted noise
            t_now: (float) current time
            t_next: (float) next time

        Returns:
            (torch.Tensor) next image
        """
        with torch.no_grad():
            gamma_now = self.get_gamma_schedule(t_now, schedule=self.noise_schedule)
            gamma_next = self.get_gamma_schedule(t_next, schedule=self.noise_schedule)
            alpha_now = gamma_now / gamma_next
            sigma_now = torch.sqrt(1 - alpha_now)

            z = torch.randn_like(x_t) if t_next > 0 else torch.zeros_like(x_t)
            z = z.to(self.device)

            pre_scale = 1 / torch.sqrt(alpha_now)
            eps_scaled = (1 - alpha_now) / torch.sqrt(1 - gamma_now)

            post_sigma = sigma_now * z

            x_previous = pre_scale * (x_t - eps_scaled * eps_pred) + post_sigma

        return x_previous

    def training_step(self, batch):
        sharp, blur = batch
        loss = self.get_loss(sharp, condition_batch=blur)
        self.log("train/loss", loss)
        wandb.log({"training loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        sharp, blur = batch

        loss = self.get_loss(sharp, condition_batch=blur)
        self.log("val/loss", loss)
        wandb.log({"validation loss": loss})
        return loss

    def test_step(self, batch, batch_idx):
        sharp, blur = batch

        # create x, a torch tensor of random noise in the same shape as the input
        reconstructed_image = torch.randn(blur.shape).to(self.device)

        steps = torch.arange(self.reverse_steps, device=self.device)

        for step in steps:
            t_now = 1 - step / self.reverse_steps
            t_next = max(1 - (step + 1) / self.reverse_steps, 0)

            # normalize the input data to have variance 1
            reconstructed_image = (
                reconstructed_image
                / torch.std(reconstructed_image, dim=[1, 2, 3], keepdim=True)
                if self.normalize
                else reconstructed_image
            )

            predicted_noise = self.forward(
                reconstructed_image,
                t_now,
                x_cond=blur,
            )

            reconstructed_image = self.ddpm_step(
                reconstructed_image, predicted_noise, t_now, t_next
            )

        # log images to wandb
        visualize_image(sharp, blur, reconstructed_image)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optimizer


def sigmoid(x):
    """Sigmoid function.

    Args:
        x: (float) input

    Returns:
        (float) out
    """
    return 1 / (1 + math.exp(-x))
