import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from skimage import restoration

cwd = os.getcwd()


# load the inference data from the inference/results folder
def load_data():
    """
    Load the inference data from the inference/results folder
    :return: the inference data
    """
    data = np.load(
        cwd
        + "/inference_results_numpy_files/denoising_autoencoder/denoising_autoencoder_full_dataset_500_epochs_10_patience_reduce_lr_on_plateau_adamw_kernel_size_3_test_data.npz"
    )
    clean_data = data["clean_data"]
    noisy_data = data["noisy_data"]

    assert clean_data.shape == noisy_data.shape

    return clean_data, noisy_data


def denoise_using_nlm(clean: np.ndarray, noisy: np.ndarray):
    """
    Denoise the data using non-local means filter
    :param clean: the clean data
    :param noisy: the noisy data
    :return: the denoised data
    """
    # use real channel
    noisy_data = noisy[:, 0, :]
    clean_data = clean[:, 0, :]

    # calculate the standard deviation of the noise
    sigma = np.std(noisy_data - clean_data)
    # Denoise the data using non-local means filter
    denoised_data = restoration.denoise_nl_means(
        noisy_data, patch_size=5, patch_distance=2, h=1.2 * sigma, fast_mode=False
    )

    # reshape the denoised data to the shape (200, 1, 512)
    denoised_data = denoised_data.reshape(200, 1, -1)

    # create array of 0s with shape as denoised_data
    array = np.zeros_like(denoised_data)

    # concatentae the array to the denoised_data along the axis 1 to get the shape (200, 2, 512)
    denoised_data = np.concatenate((denoised_data, array), axis=1)

    assert denoised_data.shape == clean.shape

    return clean, noisy, denoised_data


def save_data(clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray):
    """Save the data
    :param clean: the clean data
    :param noisy: the noisy data
    :param denoised: the denoised data
    :return: None
    """
    # save the data in the inference/results folder
    np.savez(
        cwd
        + "/inference_results_numpy_files/nlm_denoising/nlm_denoising_test_data.npz",
        clean_data=clean,
        noisy_data=noisy,
        reconstructed_data=denoised,
    )


if __name__ == "__main__":
    clean, noisy = load_data()
    clean, noisy, denoised = denoise_using_nlm(clean, noisy)
    save_data(clean, noisy, denoised)
