import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
import os
from quantification.process_and_fit_spectra import (
    process_and_save_as_nifti,
    run_fitting_process,
)
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CyclicLR,
    MultiStepLR,
    ExponentialLR,
    ReduceLROnPlateau,
)
from wandb.sdk.wandb_config import Config as wandbConfig

import wandb
from utils.utils import (
    loss_function,
    metric,
    optimizer_function,
    visualize_imaginary_transients_outputs,
    visualize_network_outputs,
    visualize_real_transients_outputs,
)


class DenoisingBaseModel(pl.LightningModule):
    def __init__(self, params: wandbConfig):
        """Base class for non-diffusion based models

        Args:
            params (wandbConfig): [description]
            transients (bool, optional): Whether the data is coil combined or transients. Defaults to False.
        """
        super().__init__()
        self.params = params
        self.transients_individually = params.transients_individually
        self.transients_conditioned = params.transients_conditioned
        # Initialize empty lists to accumulate the data
        self.clean_data_list = []
        self.noisy_data_list = []
        self.model_output_list = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Please Implement this function")

    def step(self, batch, batch_idx):
        """General step function for training, validation and testing

        Args:
            batch ([type]): A batch of data
            batch_idx ([type]): The index of the batch
            return_data (bool, optional): Whether to return the data. Defaults to False.

        Returns:
            loss: The loss value. Can be either the MSE or L1 loss.
            psnr: The Peak-Signal-To-Noise-Ratio (PSNR) value.
            mape: The Mean-Absolute-Percentage-Error (MAPE) value.
            rmse: The Root-Mean-Square-Error (RMSE) value.
            reconstructed_data/model_output: torch.Tensor of the reconstructed data or model output.
                Shape is (batch_size, number_of_channels, spectral_length).
                If the data is transients, then the shape is:
                  (batch_size, number_of_channels, number_of_transients, spectral_length).
            clean_data: The clean data. Shape is (batch_size, number_of_channels, spectral_length).
                If the data is transients, then the shape is:
                  (batch_size, number_of_channels, number_of_transients, spectral_length).
            noisy_data: The noisy data. The clean data. Shape is (batch_size, number_of_channels, spectral_length).
                If the data is transients, then the shape is:
                  (batch_size, number_of_channels, number_of_transients, spectral_length).

        """
        noisy_data, clean_data = batch

        if self.transients_individually:
            # loads the transient data of shape (batch_size, number_of_channels, number_of_transients, spectral_length)
            noisy_data = noisy_data[:, :2, :, :]
            clean_data = clean_data[:, :2, :, :]
            # assert that the shape of the data is correct
            assert (len(noisy_data.shape) == 4) and (
                len(clean_data.shape) == 4
            ), f"Shape of the data is not correct. You have to work with the transient dataset. Noisy data shape: {noisy_data.shape}, clean data shape: {clean_data.shape}"

            batch_size = noisy_data.shape[0]
            number_of_channels = noisy_data.shape[1]
            number_of_transients = noisy_data.shape[2]
            spectral_length = noisy_data.shape[3]

            # change the shape to (batch_size*number_of_transients, number_of_channels, spectral_length) by permuting the axes
            noisy_data = noisy_data.permute(0, 2, 1, 3)
            clean_data = clean_data.permute(0, 2, 1, 3)
            # change the shape to (batch_size*number_of_transients, number_of_channels, spectral_length) by using view to preserve the data
            noisy_data = noisy_data.contiguous().view(
                -1, noisy_data.shape[2], noisy_data.shape[3]
            )
            clean_data = clean_data.contiguous().view(
                -1, clean_data.shape[2], clean_data.shape[3]
            )

            # assert that the shape of the data is correct i.e. shape is (batch_size*number_of_transients, number_of_channels, spectral_length)
            assert noisy_data.shape == torch.Size(
                [
                    batch_size * number_of_transients,
                    number_of_channels,
                    spectral_length,
                ]
            ) and clean_data.shape == torch.Size(
                [
                    batch_size * number_of_transients,
                    number_of_channels,
                    spectral_length,
                ]
            ), f"Shape of the data is not correct. Noisy data shape: {noisy_data.shape}, clean data shape: {clean_data.shape}. The shape should be {(batch_size*number_of_transients, number_of_channels, spectral_length)}"
        elif self.transients_conditioned:
            # load the data of shape (batch_size, number_of_channels, spectral_length)
            noisy_data = noisy_data[:, :2, :]
            clean_data = clean_data[:, :2, :]

            batch_size = noisy_data.shape[0]

            # reshape the data to (batch_size, number_of_channels,8, spectral_length,) by repeating the data 8 times
            noisy_data = noisy_data.unsqueeze(2).repeat(1, 1, 8, 1)
            clean_data = clean_data.unsqueeze(2).repeat(1, 1, 8, 1)

            # assert that the shape of the data is correct i.e. shape is (batch_size, number_of_channels,8, spectral_length,)
            assert noisy_data.shape == torch.Size(
                [batch_size, 2, 8, 512]
            ) and clean_data.shape == torch.Size(
                [batch_size, 2, 8, 512]
            ), f"Shape of the data is not correct. Noisy data shape: {noisy_data.shape}, clean data shape: {clean_data.shape}. The shape should be {(batch_size, number_of_channels,8, spectral_length,)}"

            # only real and imaginary channels are used when training the model
            noisy_data = noisy_data[:, :2, :, :]
            clean_data = clean_data[:, :2, :, :]

        else:
            noisy_data = noisy_data[:, :2, :]
            clean_data = clean_data[:, :2, :]

        noise = noisy_data - clean_data

        model_ouput = self(noisy_data)
        # assert that reconstructed data, noisy data and clean data have the same shape
        assert (
            model_ouput.shape == noisy_data.shape == clean_data.shape
        ), f"Reconstructed data shape: {model_ouput.shape}, noisy data shape: {noisy_data.shape}, clean data shape: {clean_data.shape}"

        if self.params.predict_noise:
            loss = loss_function(model_ouput, noise, self.params.loss)
            reconstructed_data = noisy_data - model_ouput
            psnr = metric(reconstructed_data, clean_data, self.device)["psnr"]
            mape = metric(reconstructed_data, clean_data, self.device)["mape"]
            rmse = metric(reconstructed_data, clean_data, self.device)["rmse"]
        else:
            loss = loss_function(model_ouput, clean_data, self.params.loss)
            psnr = metric(model_ouput, clean_data, self.device)["psnr"]
            mape = metric(model_ouput, clean_data, self.device)["mape"]
            rmse = metric(model_ouput, clean_data, self.device)["rmse"]

        if self.params.predict_noise:
            return loss, psnr, mape, rmse, reconstructed_data, clean_data, noisy_data
        else:
            return loss, psnr, mape, rmse, model_ouput, clean_data, noisy_data

    def training_step(self, batch, batch_idx):
        loss, psnr, mape, rmse, model_output, clean_data, noisy_data = self.step(
            batch, batch_idx
        )

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_psnr", psnr)
        self.log("train_mape", mape)
        self.log("train_rmse", rmse)

        return {
            "loss": loss,
            "psnr": psnr,
            "mape": mape,
            "rmse": rmse,
        }

    def validation_step(self, batch, batch_idx):
        loss, psnr, mape, rmse, model_ouput, clean_data, noisy_data = self.step(
            batch, batch_idx
        )

        if self.transients_individually:
            # change the shape to (batch_size, number_of_transients, number_of_channels, spectral_length) by using view to preserve the data
            noisy_data = noisy_data.view(
                -1,
                self.params.number_of_transients,
                noisy_data.shape[1],
                noisy_data.shape[2],
            )
            clean_data = clean_data.view(
                -1,
                self.params.number_of_transients,
                clean_data.shape[1],
                clean_data.shape[2],
            )
            model_ouput = model_ouput.view(
                -1,
                self.params.number_of_transients,
                model_ouput.shape[1],
                model_ouput.shape[2],
            )

            # change the shape to (batch_size, number_of_transients, number_of_channels, spectral_length) by permuting the axes
            noisy_data = noisy_data.permute(0, 2, 1, 3)
            clean_data = clean_data.permute(0, 2, 1, 3)
            model_ouput = model_ouput.permute(0, 2, 1, 3)

            # average over the number of transients
            clean_data = torch.mean(clean_data, axis=2)
            noisy_data = torch.mean(noisy_data, axis=2)
            model_ouput = torch.mean(model_ouput, axis=2)

        elif self.transients_conditioned:
            # average over the number of transients
            clean_data = torch.mean(clean_data, axis=2)
            noisy_data = torch.mean(noisy_data, axis=2)
            model_ouput = torch.mean(model_ouput, axis=2)

        visualize_network_outputs(model_ouput, clean_data, noisy_data, 2)

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
        loss, psnr, mape, rmse, model_ouput, clean_data, noisy_data = self.step(
            batch, batch_idx
        )

        if self.transients_individually:
            # change the shape to (batch_size, number_of_transients, number_of_channels, spectral_length) by using view to preserve the data
            noisy_data = noisy_data.view(
                -1,
                self.params.number_of_transients,
                noisy_data.shape[1],
                noisy_data.shape[2],
            )
            clean_data = clean_data.view(
                -1,
                self.params.number_of_transients,
                clean_data.shape[1],
                clean_data.shape[2],
            )
            model_ouput = model_ouput.view(
                -1,
                self.params.number_of_transients,
                model_ouput.shape[1],
                model_ouput.shape[2],
            )

            # change the shape to (batch_size, number_of_transients, number_of_channels, spectral_length) by permuting the axes
            noisy_data = noisy_data.permute(0, 2, 1, 3)
            clean_data = clean_data.permute(0, 2, 1, 3)
            model_ouput = model_ouput.permute(0, 2, 1, 3)

            # average over the number of transients
            clean_data = torch.mean(clean_data, axis=2)
            noisy_data = torch.mean(noisy_data, axis=2)
            model_ouput = torch.mean(model_ouput, axis=2)

        elif self.transients_conditioned:
            # average over the number of transients
            clean_data = torch.mean(clean_data, axis=2)
            noisy_data = torch.mean(noisy_data, axis=2)
            model_ouput = torch.mean(model_ouput, axis=2)

        visualize_network_outputs(model_ouput, clean_data, noisy_data, 2)

        # Accumulate the data in a list
        self.clean_data_list.append(clean_data.cpu().numpy())
        self.noisy_data_list.append(noisy_data.cpu().numpy())
        self.model_output_list.append(model_ouput.cpu().numpy())

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_psnr", psnr)
        self.log("test_mape", mape)
        self.log("test_rmse", rmse)

        return {
            "loss": loss,
            "psnr": psnr,
            "mape": mape,
            "rmse": rmse,
        }

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
        optimizer = optimizer_function(self.parameters(), self.params)
        if self.params.lr_scheduler == "multi_step":
            scheduler = MultiStepLR(
                optimizer, milestones=[200, 400, 800], gamma=self.params.lr_decay
            )
            return [optimizer], [scheduler]
        elif self.params.lr_scheduler == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.params.T_max)
            return [optimizer], [scheduler]
        elif self.params.lr_scheduler == "cyclic":
            scheduler = CyclicLR(
                optimizer,
                base_lr=self.params.lr,
                max_lr=self.params.lr * 10,
                step_size_up=200,
                step_size_down=200,
                mode="triangular2",
            )
            return [optimizer], [scheduler]
        elif self.params.lr_scheduler == "exponential":
            scheduler = ExponentialLR(
                optimizer, gamma=self.params.lr_decay, last_epoch=-1
            )
            return [optimizer], [scheduler]
        elif self.params.lr_scheduler == "reduce_on_plateau":
            optimizer = optimizer_function(self.parameters(), self.params)
            lr_scheduler = {
                "scheduler": ReduceLROnPlateau(
                    optimizer, mode="min", patience=3, verbose=True, min_lr=1e-6
                ),
                "monitor": "val_loss",
                "frequency": 10  # Frequency should be an integer, not a string
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            }
            return [optimizer], [lr_scheduler]

        else:
            # Default is no scheduler i.e. constant lr

            return optimizer
