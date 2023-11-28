import math
from abc import ABC, abstractmethod

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from wandb.sdk.wandb_config import Config as wandbConfig
import os

import wandb
from utils.utils import (
    loss_function,
    metric,
    optimizer_function,
    visualize_network_outputs,
)


class DiffusionBaseModel(pl.LightningModule):
    """Architecture of the base diffusion model.

    This base class is used to create all the functions needed to train
    models using diffusion as described in the paper: Denoising Diffusion
    Probabilistic Models by Ho et al. (2020). A link to the paper can be
    found here: https://arxiv.org/pdf/2006.11239.pdf

    The code has been adapted from the official implementation of the paper

    Attributes:
        params: (wandbConfig) parameters for the model
        input_size: (int) size of the input data or spectral length
        diffusion_steps: (int) number of diffusion steps
        input_channels: (int) number of channels in the input data
        output_channels: (int) number of channels in the output data
        beta_small: (float) lower threshold beta
        beta_large: (float) upper threshold beta
        beta_offest: (float) offset for the beta value
        diffusion_beta_schedule: (str) type of beta schedule. Can be one of
            linear, original, cosine or cosine_openai
        conditional: (bool) whether the model is conditional or not.
            The conditioning mechanism is described in the follow-up paper:
            Saharia et al. "Image super-resolution via iterative refinement."
            arXiv preprint arXiv:2103.16774 (2021).
        loss: (str) type of loss function. Can be one of mse, l1
        transients: (bool) whether the model is trained on transients or coil-combined data


    """

    def __init__(self, params: wandbConfig):
        """Inits DiffusionBaseModel with the parameters from the config file.

        Args:
            params: (wandbConfig) parameters for the model

        """
        super().__init__()
        self.params = params
        self.input_size = params.input_size
        self.diffusion_steps = params.diffusion_steps
        self.beta_small = params.beta_small
        self.beta_large = params.beta_large
        self.beta_offset = params.beta_offset  # OpenAI use 1/157.5 as offset
        self.diffusion_beta_schedule = params.diffusion_beta_schedule
        self.conditional = params.conditional
        self.loss = params.loss
        self.transients = params.transients

        # Initialize empty lists to accumulate the data
        self.clean_data_list = []
        self.noisy_data_list = []
        self.model_output_list = []

    @abstractmethod
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, x_condition=None
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: (torch.Tensor) input data with noise added according to timestep
            t: (torch.Tensor) Tensor of diffusion time steps
            x_condition: (torch.Tensor, optional) conditioning data. Defaults to None.

        Returns:
            (torch.Tensor) output of the forward pass

        """
        raise NotImplementedError(
            "Every model needs to have a forward pass. Please Implement this function"
        )

    def beta(self, t, schedule) -> float:
        """Calculates the beta value for the given time step.

        Args:
            t: (int) diffusion time step
            schedule: (str) type of beta schedule. Can be one of
                linear, original, cosine or cosine_openai

        Returns:
            (float) beta value for the given time step

        """
        if schedule == "linear":
            return self.linear_beta(t)
        elif schedule == "original":
            # //NOTE: Used in the original Lighnting repo.
            # //  I think this is wrong because it doesn't work if you change
            # //  the number of diffusion steps.
            return self.beta_small + (t / self.t_range) * (
                self.beta_large - self.beta_small
            )
        elif schedule == "cosine":
            return self.cosine_beta(t)
        elif schedule == "cosine_openai":
            return (
                self.cosine_beta_openai(
                    t,
                    alpha_bar=lambda t: math.cos(
                        (t + self.beta_offset) / (1 + self.beta_offset)
                    )
                    * math.pi
                    / 2,
                )
                ** 2
            )
        else:
            raise ValueError("Unknown beta schedule")

    def linear_beta(self, t) -> float:
        """Linear beta schedule as implemented by lucidrains.

        Calculates the beta value for a linear schedule.
        The code has been adapted from lucidrains' implementation of denoising diffusion.
        The link to the original code is: //TODO: add link

        Args:
            t: (int) diffusion time step

        Returns:
            (float) beta value for the given time step

        """
        scale = 1000 / self.diffusion_steps
        beta_start = scale * self.beta_small  # 0.0001
        beta_end = scale * self.beta_large  # 0.02
        linear_betas = np.linspace(beta_start, beta_end, self.diffusion_steps)
        return linear_betas[t]

    def cosine_beta(self, t) -> float:
        """Implementation of the cosine beta schedule by Lucidrains.


        Calculates the beta value for a cosine schedule.
        The code has been adapted from lucidrains' implementation of denoising diffusion.
        The link to the original code is: //TODO: add link

        Args:
            t: (int) diffusion time step

        Returns:
            (float) beta value for the given time step

        """
        steps = self.diffusion_steps + 1
        x = np.linspace(0, self.diffusion_steps, steps)
        alphas_cumprod = (
            np.cos(
                ((x / self.diffusion_steps) + self.beta_offset)
                / (1 + self.beta_offset)
                * math.pi
                * 0.5
            )
            ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        clipped_betas = np.clip(betas, 0, 0.999)
        return clipped_betas[t]

    def cosine_beta_openai(self, t, alpha_bar, max_beta=0.999):
        """Implementation of the cosine beta schedule by OpenAI.

        Create a beta schedule that discretizes the given alpha_t_bar function,
        which defines the cumulative product of (1-beta) over time from t = [0,1].

        Args:
            t: (int) diffusion time step
            alpha_bar: a lambda that takes an argument t from 0 to 1 and
                        produces the cumulative product of (1-beta) up to that
                        part of the diffusion process.
            max_beta: the maximum beta to use; use values lower than 1 to
                        prevent singularities.
        """
        betas = []
        for i in range(self.diffusion_steps):
            t1 = i / self.diffusion_steps
            t2 = (i + 1) / self.diffusion_steps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)[t]

    def alpha(self, t) -> float:
        """Calculates the alpha value for the given time step.

        Args:
            t: (int) diffusion time step

        Returns:
            (float) alpha value for the given time step

        """
        return 1 - self.beta(t, self.diffusion_beta_schedule)

    def alpha_bar(self, t):
        """Calculates alpha_bar or the cumulative sum of alpha for all timesteps.

        Args:
            t: (int) diffusion time step

        Returns:
            (float) alpha_bar value for the given time step

        """
        return math.prod([self.alpha(j) for j in range(t)])

    # //TODO: create a new function to calculate the noise level according to the Wavegrad paper

    def get_loss(self, batch, condition_batch=None):
        """
        Corresponds to Algorithm 1 from (Ho et al., 2020).
        """
        ts = torch.randint(
            0, self.diffusion_steps, [batch.shape[0]], device=self.device
        )
        noise_imgs = []
        epsilons = torch.randn(batch.shape, device=self.device)
        for i in range(len(ts)):
            a_hat = self.alpha_bar(ts[i])
            noise_imgs.append(
                (math.sqrt(a_hat) * batch[i]) + (math.sqrt(1 - a_hat) * epsilons[i])
            )
        noise_imgs = torch.stack(noise_imgs, dim=0)
        if condition_batch is not None:
            e_hat = self.forward(
                noise_imgs, ts.unsqueeze(-1).type(torch.float), condition_batch
            )

        else:
            e_hat = self.forward(noise_imgs, ts.unsqueeze(-1).type(torch.float))

        if self.loss == "l1":
            loss = nn.functional.l1_loss(
                e_hat.reshape(-1, self.input_size),
                epsilons.reshape(-1, self.input_size),
            )
        elif self.loss == "mse":
            loss = nn.functional.mse_loss(
                e_hat.reshape(-1, self.input_size),
                epsilons.reshape(-1, self.input_size),
            )
        else:
            raise ValueError("Invalid loss type")
        return loss

    def denoise_sample(self, x, t, x_cond=None):
        """
        Corresponds to the inner loop of Algorithm 2 from (Ho et al., 2020).
        """
        with torch.no_grad():
            if t > 1:
                z = torch.randn(x.shape).to(self.device)
            else:
                z = torch.zeros_like(x).to(self.device)
            if x_cond is not None:
                e_hat = self.forward(x, t.view(1, 1).repeat(x.shape[0], 1), x_cond).to(
                    self.device
                )
            else:
                e_hat = self.forward(x, t.view(1, 1).repeat(x.shape[0], 1)).to(
                    self.device
                )

            pre_scale = 1 / math.sqrt(self.alpha(t))
            e_scale = (1 - self.alpha(t)) / math.sqrt(1 - self.alpha_bar(t))
            post_sigma = (math.sqrt(self.beta(t, self.diffusion_beta_schedule)) * z).to(
                self.device
            )
            x = pre_scale * (x - e_scale * e_hat) + post_sigma
            return x

    def training_step(self, batch):
        noisy_data, clean_data = batch
        if self.transients:
            noisy_data = noisy_data[:, :2, :, :]
            clean_data = clean_data[:, :2, :, :]
        else:
            noisy_data = noisy_data[:, :2, :]
            clean_data = clean_data[:, :2, :]

        loss = self.get_loss(clean_data, condition_batch=noisy_data)
        self.log("train/loss", loss)
        wandb.log({"training loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        noisy_data, clean_data = batch

        if self.transients:
            noisy_data = noisy_data[:, :2, :, :]
            clean_data = clean_data[:, :2, :, :]
        else:
            noisy_data = noisy_data[:, :2, :]
            clean_data = clean_data[:, :2, :]

        loss = self.get_loss(clean_data, condition_batch=noisy_data)
        # create x, a torch tensor of random noise in the same shape as the input
        x = torch.randn(noisy_data.shape).to(self.device)

        sample_steps = torch.arange(self.diffusion_steps - 1, 0, -1).to(self.device)

        for t in sample_steps:
            x = self.denoise_sample(x, t, noisy_data)

        visualize_network_outputs(x, clean_data, noisy_data, 2)
        psnr = metric(x, clean_data, self.device)["psnr"]
        mape = metric(x, clean_data, self.device)["mape"]
        rmse = metric(x, clean_data, self.device)["rmse"]

        self.log("val_loss", loss, prog_bar=True)
        self.log("vals_psnr", psnr)
        self.log("vals_mape", mape)
        self.log("vals_rmse", rmse)
        return {
            "loss": loss,
            "psnr": psnr,
            "mape": mape,
            "rmse": rmse,
        }

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        noisy_data, clean_data = batch

        if self.transients:
            noisy_data = noisy_data[:, :2, :, :]
            clean_data = clean_data[:, :2, :, :]
        else:
            noisy_data = noisy_data[:, :2, :]
            clean_data = clean_data[:, :2, :]

        # create x, a torch tensor of random noise in the same shape as the input
        x = torch.randn(noisy_data.shape).to(self.device)

        sample_steps = torch.arange(self.diffusion_steps - 1, 0, -1).to(self.device)

        for t in sample_steps:
            x = self.denoise_sample(x, t, noisy_data)
            psnr = metric(x, clean_data, device=self.device)["psnr"]
            wandb.log({f"test psnr at step {t}": psnr})

        visualize_network_outputs(x, clean_data, noisy_data, 2)
        self.clean_data_list.append(clean_data.cpu().numpy())
        self.noisy_data_list.append(noisy_data.cpu().numpy())
        self.model_output_list.append(x.cpu().numpy())

        psnr = metric(x, clean_data, device=self.device)["psnr"]
        wandb.log({"test psnr": psnr})

    @torch.no_grad()
    def test_epoch_end(self, outputs):
        # Concatenate the accumulated data into arrays
        clean_data_array = np.concatenate(self.clean_data_list, axis=0)
        noisy_data_array = np.concatenate(self.noisy_data_list, axis=0)
        reconstructed_output_array = np.concatenate(self.model_output_list, axis=0)

        folder_path = self.params.save_test_results_in_folder
        file_path = os.path.join(folder_path, self.params.run_name + "_test_data.npz")

        # Save the arrays to a .npz file
        np.savez(
            file=file_path,
            clean_data=clean_data_array,
            noisy_data=noisy_data_array,
            reconstructed_data=reconstructed_output_array,
        )

        # Clear the lists for the next test run
        self.clean_data_list = []
        self.noisy_data_list = []
        self.model_output_list = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-4)
        return optimizer
