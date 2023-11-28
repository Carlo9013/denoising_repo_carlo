"""Primary denoising functions for low rank denoising
Copyright Will Clarke, University of Oxford, 2021"""

# add utils to path
import os
import sys
from itertools import combinations

import numpy as np

"""Utility functions for low rank denoising
Copyright Will Clarke, University of Oxford, 2021"""
cwd = os.getcwd()


def lsvd(A, r=None, sv_lim=None):
    """Calculate the svd decomposition of A.
    Truncated to rank r or to a SV limit (e.g. determined by MP)
    Copied from Mark Chiew's Matlab code.

    U, S, V = lsvd(A, r)

    U, V are the left and right singular vectors of A
    S contains the singular values

    A is a real or complex input matrix, where size(A,1) > size(A,2)
    r is the desired output rank of U*S*V' where 1 <= r <= size(A,2)

    :param A: Input matrix
    :type A: numpy.ndarray
    :param r: Truncate to rank r, defaults to None (no truncation)
    :type r: int, optional
    :param sv_lim: Singular value threshold, used if r is None.
        Defaults to None
    :type sv_lim: float, optional
    :return: Truncated estimate of A
    :rtype: numpy.ndarray
    """
    # Quick checks
    if r is not None and r < 1:
        raise ValueError("Rank must be > 0")
    if sv_lim is not None and sv_lim < 0.0:
        raise ValueError("sv_lim must be > 0.0")

    if A.shape[1] > A.shape[0]:
        swap = True
        A = A.conj().T
    else:
        swap = False

    D, V = np.linalg.eigh(A.conj().T @ A)
    D[D < 0.0] = 1e-15
    S = D**0.5

    if r is None and sv_lim is None:
        # No truncation
        r = A.shape[1]
    elif r is None and sv_lim > 0.0:
        r = len(S) - np.searchsorted(S, sv_lim)
        if r == 0:
            r = 1
            # raise ValueError('MP limit higher than any singular value.')
        # print(f'Truncate to r = {r}')

    S = S[-r:]
    V = V[:, -r:]
    U = A @ (V @ np.diag(1 / S))
    S = np.diag(S)

    if swap:
        return V, S, U
    else:
        return U, S, V


def svd_trunc(A, r=None, sv_lim=None):
    """SVD based truncation of A to rank r, or SV threshold sv_lim.
    A may have more than 2 dimensions, but it will be
    reshaped to two dimensions, preserving the length of the
    final dimension.

    Also returns estimates of the (unscaled) variance and covariance of the
    truncated matrix A.

    :param A: Input matrix
    :type A: numpy.ndarray
    :param r: Truncate to rank r, defaults to None (no truncation)
    :type r: int, optional
    :param sv_lim: Singular value threshold, used if r is None.
        Defaults to None
    :type sv_lim: float, optional
    :return: Truncated estimate of A
    :rtype: numpy.ndarray
    :return: Estimate of the variance in A
    :rtype: numpy.ndarray
    :return: Estimate of the covariance in the last dimension of A.
    :rtype: numpy.ndarray
    """
    dims = A.shape
    u, s, v = lsvd(A.reshape((-1, dims[-1])), r=r, sv_lim=sv_lim)

    est_var = calc_var(u, v)
    est_covar = calc_covar(v)

    return (u @ s @ v.conj().T).reshape(dims), est_var.reshape(dims), est_covar


def calc_var(u, v):
    """Calculate the unscaled variance from a truncated SVD.
    Uses the method proposed in Chen Y, Fan J, Ma C, Yan Y.
    Inference and uncertainty quantification for noisy matrix completion.
    PNAS 2019;116:22931–22937 doi: 10.1073/pnas.1910053116.

    :param u: Left singular vectors of M
    :type u: numpy.ndarray
    :param v: Right singular vectors of M
    :type v: numpy.ndarray
    :return: Estimated element-wise variance of M
    :rtype: numpy.ndarray
    """
    return (np.linalg.norm(u, axis=1) ** 2)[:, np.newaxis] + np.linalg.norm(
        v, axis=1
    ) ** 2


def calc_covar(u):
    """Calculate the estimated unscaled off-diagonal covariance
    from a truncated SVD.

    :param u: Left or right singular vectors of M
    :type u: numpy.ndarray
    :return: Estimated unscaled off-diagonal covariance of M
    :rtype: numpy.ndarray
    """
    return u @ u.conj().T


def max_sv_from_mp(data_var, data_shape):
    """Calculate the upper Marchenko–Pastur limit for a pure noise
    Matrix of defined shape and data variance.

    :param data_var: Noise variance
    :type data_var: float
    :param data_shape: 2-tuple of data dimensions.
    :type data_shape: tuple
    :return: Upper MP singular value limit
    :rtype: float
    """
    # Make sure dimensions agree with utils.lsvd
    # utils.lsvd will always move largest dimension to first dim.
    if data_shape[1] > data_shape[0]:
        data_shape = (data_shape[1], data_shape[0])

    c = data_shape[1] / data_shape[0]
    sv_lim = data_var * (1 + np.sqrt(c)) ** 2
    return (sv_lim * data_shape[0]) ** 0.5


def st_denoising(noisy_data, rank=None, mp_var=None, patch=None, step=1):
    """Apply ST denoising truncating the data to rank r, or upto MP defined
    limit. Can be applied globally or locally (patch-wise).

    Incorporates estimation of uncertainty adapted from
    Uncertainty Quantification for Hyperspectral Image Denoising Frameworks
    based on Low-rank Matrix Approximation
    Shaobo Xia, Jingwei Song, Dong Chen, Jun Wang
    arXiv:2004.10959

    :param noisy_data:  Array of data of shape NXxNYxNZx...xNt
        (time domain last).
    :type noisy_data: np.ndarray
    :param rank: Truncate to rank r, defaults to None (no truncation). If set
        to 'MP' then truncation will be set using the limit calculated using
        the mp_var argument.
    :type rank: int or str, optional
    :param mp_var: Variance to use to estimate the MP derived truncation.
    :type mp_var: float, optional
    :param patch: 3D patch size, defaults to None for global.
    :type patch: tuple, optional
    :param step: Patch step size, defaults to 1
    :type step: int, optional
    :return: Denoised data
    :rtype: np.ndarray
    :return: Estimated variance
    :rtype: np.ndarray
    :return: Estimated covariance in final (time) dimension
    :rtype: np.ndarray
    """

    truncated_data = np.zeros_like(noisy_data)
    if patch:
        if rank is None or isinstance(rank, int):

            def svd_t(x):
                return svd_trunc(x, r=rank)

        elif isinstance(rank, str) and rank.lower() == "mp" and mp_var is not None:
            mp_shape = (
                np.prod(tuple(patch) + noisy_data.shape[3:-1]),
                noisy_data.shape[-1],
            )
            mp_lim = max_sv_from_mp(mp_var, mp_shape)

            def svd_t(x):
                return svd_trunc(x, sv_lim=mp_lim)

        else:
            raise TypeError(
                'rank parameter must be positive integer or "mp".'
                ' If "mp" then mp_var must be specified.'
            )

        data_shape = noisy_data.shape[:3]
        M = np.full(data_shape, False)
        M[::step, ::step, ::step] = True
        ii, jj, kk = np.where(M)

        idx_i = np.arange(0, np.min((patch[0], data_shape[0])))
        idx_j = np.arange(0, np.min((patch[1], data_shape[1])))
        idx_k = np.arange(0, np.min((patch[2], data_shape[2])))

        navg = np.zeros((len(ii),) + data_shape)
        covar_out = []
        variance_patches = np.zeros((len(ii),) + noisy_data.shape)
        for count, (idx, jdx, kdx) in enumerate(zip(ii, jj, kk)):
            i = np.mod(idx + idx_i - 1, data_shape[0])
            j = np.mod(jdx + idx_j - 1, data_shape[1])
            k = np.mod(kdx + idx_k - 1, data_shape[2])

            denoised, var, cvar = svd_t(noisy_data[np.ix_(i, j, k)])
            truncated_data[np.ix_(i, j, k)] += denoised
            variance_patches[np.ix_([count], i, j, k)] = var
            covar_out.append(cvar)
            navg[np.ix_([count], i, j, k)] += 1

        navg_all = np.sum(navg, axis=0)
        # Deal with broadcasting comparing along trailing dimension
        truncated_data = (truncated_data.T / navg_all.T).T

        variance = calculate_patch_variance(variance_patches, navg, patch)

        covar = np.asarray(covar_out).mean(axis=0)
    else:
        # Default to global
        if rank is None or isinstance(rank, int):
            truncated_data, variance, covar = svd_trunc(noisy_data, r=rank)
        elif isinstance(rank, str) and rank.lower() == "mp" and mp_var is not None:
            mp_shape = (np.prod(noisy_data.shape[:-1]), noisy_data.shape[-1])
            mp_lim = max_sv_from_mp(mp_var, mp_shape)
            truncated_data, variance, covar = svd_trunc(noisy_data, sv_lim=mp_lim)
        else:
            raise TypeError(
                'rank parameter must be positive integer or "mp".'
                ' If "mp" then mp_var must be specified.'
            )

    return truncated_data, variance, covar


def calculate_patch_variance(patch_var, patch_positions, patch_size):
    """Function to calculate patch-wise variance

    Based on Uncertainty Quantification for Hyperspectral Image Denoising
    Frameworks based on Low-rank Matrix Approximation
    Shaobo Xia, Jingwei Song, Dong Chen, Jun Wang
    arXiv:2004.10959

    :param patch_var: Array of variance with first dimension
        containing patches
    :type patch_var: numpy.ndarray
    :param patch_positions: Array of patch positions with first
        dimension containing patches
    :type patch_positions: numpy.ndarray
    :param patch_size: 3D patch size
    :type patch_size: tuple
    :return: Estimated variance
    :rtype: numpy.ndarray
    """
    # Number of patches in each position
    npatch_all = np.sum(patch_positions, axis=0)

    # Calculate the first (self) term of the variance
    var_self = (np.sum(patch_var, axis=0).T / npatch_all.T**2).T

    # Calculate the cross term
    # Calculate the overlap of each patch with each other
    # Equivalent to prameter nu^{(ijk)(i'j'k')}
    overlap_mat = np.zeros((len(patch_positions), len(patch_positions)))
    for idx, ppi in enumerate(patch_positions):
        for jdx, ppj in enumerate(patch_positions):
            overlap_mat[idx, jdx] = np.sum(ppi * ppj) / np.prod(patch_size)

    # Loop over all voxels and sum up the cross terms
    var_cross = np.zeros_like(patch_var[0])
    for ind in np.ndindex(npatch_all.shape):
        vox_patches = patch_positions[:, ind[0], ind[1], ind[2]].nonzero()[0]
        for comb in combinations(vox_patches, 2):
            var_cross[ind[0], ind[1], ind[2], ...] += (
                2
                * overlap_mat[comb[0], comb[1]]
                * np.sqrt(patch_var[comb[0], ind[0], ind[1], ind[2], ...])
                * np.sqrt(patch_var[comb[1], ind[0], ind[1], ind[2], ...])
            )

    # Normalise the cross terms
    var_cross = (var_cross.T / (npatch_all.T) ** 2).T

    return var_self + var_cross


def lp(data, r, W=None):
    """Run linear-prediction denoising on 3D MRSI data.
    :param np.ndarray data: space-time MRSI data
                      assume dimensions [Nx, Ny, Nz, ..., Nt]
    :param int r: rank threshold for Hankel filtering constraint
    :param int W: Size of 1st Hankel matrix dimension.
        If None, defaults to half of time domain length.
    """
    data_shape = data.shape
    out = data.copy().reshape(-1, data_shape[-1])

    # Loop over all the FIDs and apply
    # Hankel matrix Low-Rank enforcement
    if W is None:
        W = int(out.shape[1] / 2)
    elif W >= out.shape[1]:
        raise ValueError("W must be smaller than the time domain length.")
    K = out.shape[1] - W
    H = np.zeros((W, W + 1), dtype=complex)
    for idx, v_data in enumerate(out):
        for wdx in range(W):
            H[wdx, :] = v_data[wdx : (wdx + K + 1)]
        H, _, _ = svd_trunc(H, r=r)

        out[idx, :] = np.concatenate((H[0, :], H[1:, -1].T))

    return out.reshape(data_shape)


def lora(data, r1, r2, r1_mp_var=None, W=None):
    """Run LORA denoising on 3D MRSI data.
    :param np.ndarray data: space-time MRSI data
                      assume dimensions [Nx, Ny, Nz, ..., Nt]
    :param int r1: rank threshold for space-time LR constraint.
        Set to 'mp' for Marchenko–Pastur threshold.
    :param int r2: rank threshold for Hankel filtering constraint
    :param float r1_mp_var: Variance of noise in input data
    :param int W: Size of 1st Hankel matrix dimension.
        If None, defaults to half of time domain length.
    """

    if r1 is None or isinstance(r1, int):

        def svd_t(x):
            return svd_trunc(x, r=r1)

    elif isinstance(r1, str) and r1.lower() == "mp" and r1_mp_var is not None:
        mp_shape = (np.prod(data.shape[:-1]), data.shape[-1])
        mp_lim = max_sv_from_mp(r1_mp_var, mp_shape)

        def svd_t(x):
            return svd_trunc(x, sv_lim=mp_lim)

    else:
        raise TypeError(
            'r1 parameter must be positive integer or "mp".'
            ' If "mp" then r1_mp_var must be specified.'
        )

    data_shape = data.shape
    out = data.copy().reshape(-1, data_shape[-1])

    # Space-time low rank enforcement
    out = out.T
    out, _, _ = svd_t(out)
    out = out.T

    # Loop over all the FIDs and apply
    # Hankel matrix Low-Rank enforcement
    if W is None:
        W = int(out.shape[1] / 2)
    elif W >= out.shape[1]:
        raise ValueError("W must be smaller than the time domain length.")
    K = out.shape[1] - W
    H = np.zeros((W, W + 1), dtype=complex)
    for idx, v_data in enumerate(out):
        for wdx in range(W):
            H[wdx, :] = v_data[wdx : (wdx + K + 1)]
        H, _, _ = svd_trunc(H, r=r2)

        out[idx, :] = np.concatenate((H[0, :], H[1:, -1].T))

    return out.reshape(data_shape)


def llora(data, r1, r2, patch, step=1, r1_mp_var=None, W=None):
    """Run local-LORA denoising on 3D MRSI data.
    :param np.ndarray data: space-time MRSI data
                      assume dimensions [Nx, Ny, Nz, ..., Nt]
    :param int r1: rank threshold for space-time LR constraint
    :param int r2: rank threshold for Hankel filtering constraint
    :param patch: Spatial patch size, 3-tuple of patch in X,Y,Z.
    :param step: Size of patch step.
    :param float r1_mp_var: Variance of noise in input data
    :param int W: Size of 1st Hankel matrix dimension.
        If None, defaults to half of time domain length.
    """
    data_shape = data.shape[:3]
    M = np.full(data_shape, False)
    M[::step, ::step, ::step] = True
    ii, jj, kk = np.where(M)

    idx_i = np.arange(0, np.min((patch[0], data_shape[0])))
    idx_j = np.arange(0, np.min((patch[1], data_shape[1])))
    idx_k = np.arange(0, np.min((patch[2], data_shape[2])))

    navg = np.zeros((len(ii),) + data_shape)
    out = np.zeros_like(data)
    for count, (idx, jdx, kdx) in enumerate(zip(ii, jj, kk)):
        i = np.mod(idx + idx_i - 1, data_shape[0])
        j = np.mod(jdx + idx_j - 1, data_shape[1])
        k = np.mod(kdx + idx_k - 1, data_shape[2])

        out[np.ix_(i, j, k)] += lora(
            data[np.ix_(i, j, k)], r1, r2, r1_mp_var=r1_mp_var, W=W
        )

        navg[np.ix_([count], i, j, k)] += 1

    navg_all = np.sum(navg, axis=0)
    # Deal with broadcasting comparing along trailing dimension
    out = (out.T / navg_all.T).T

    return out


def sure_svt(data, var, patch, step=1):
    """Run SURE optimised SVT denoising on 3D MRSI data.
    :param np.ndarray data: space-time MRSI data
                      assume dimensions [Nx, Ny, Nz, ..., Nt]
    :param float var: Variance of real / imag noise in input data.
         I.e. np.var(complex_noise)/2 == np.var(np.real(complex_noise))
    :param patch: Spatial patch size, 3-tuple of patch in X,Y,Z.
    :param step: Size of patch step.
    """

    def div_SVT(lam, S, Nx, Nt):
        z = (2 * np.abs(Nx - Nt) + 1) * np.sum(np.maximum(1 - lam / S, 0)) + np.sum(
            S > lam
        )

        ss = np.subtract.outer(S**2, S**2)
        ss = ss[~np.eye(S.shape[0], dtype=bool)].reshape(S.shape[0], -1)
        z += np.sum(4 * S * np.maximum(S - lam, 0) / ss.T)

        return z

    def SURE(lam, S, v, Nx, Nt):
        return (
            -2 * Nx * Nt * v
            + np.sum(np.minimum(S**2, lam**2))
            + 2 * v * div_SVT(lam, S, Nx, Nt)
        )

    data_shape = data.shape[:3]
    M = np.full(data_shape, False)
    M[::step, ::step, ::step] = True
    ii, jj, kk = np.where(M)

    idx_i = np.arange(0, np.min((patch[0], data_shape[0])))
    idx_j = np.arange(0, np.min((patch[1], data_shape[1])))
    idx_k = np.arange(0, np.min((patch[2], data_shape[2])))

    navg = np.zeros((len(ii),) + data_shape)
    out = np.zeros_like(data)
    for count, (idx, jdx, kdx) in enumerate(zip(ii, jj, kk)):
        i = np.mod(idx + idx_i - 1, data_shape[0])
        j = np.mod(jdx + idx_j - 1, data_shape[1])
        k = np.mod(kdx + idx_k - 1, data_shape[2])

        curr_data = data[np.ix_(i, j, k)].reshape(-1, data[np.ix_(i, j, k)].shape[-1])

        U, S, V = lsvd(curr_data)
        S = np.diag(S)

        # Search up to second highest singular value.
        q = np.logspace(np.log10(S[-2]), np.log10(S[0]), num=50)
        w = np.zeros((50,))

        for idx in range(50):
            w[idx] = SURE(q[idx], S, var, curr_data.shape[0], curr_data.shape[-1])

        thresh_SURE = q[np.argmin(w)]
        # breakpoint()
        S_scaled = np.diag(
            np.maximum(S - thresh_SURE, 0) * S[-1] / (S[-1] - thresh_SURE)
        )
        out[np.ix_(i, j, k)] += (U @ S_scaled @ V.conj().T).reshape(
            data[np.ix_(i, j, k)].shape
        )

        navg[np.ix_([count], i, j, k)] += 1

    navg_all = np.sum(navg, axis=0)
    # Deal with broadcasting comparing along trailing dimension
    out = (out.T / navg_all.T).T

    return out


def sure_hard(data, var, patch, step=1):
    """Run SURE optimised SVT denoising on 3D MRSI data.
    :param np.ndarray data: space-time MRSI data
                      assume dimensions [Nx, Ny, Nz, ..., Nt]
    :param float var: Variance of real / imag noise in input data.
         I.e. np.var(complex_noise)/2 == np.var(np.real(complex_noise))
    :param patch: Spatial patch size, 3-tuple of patch in X,Y,Z.
    :param step: Size of patch step.
    """

    def div_HARD(r, S, Nx, Nt):
        z = (Nx + Nt) * r - r**2
        for idx in range(r):
            for jdx in range(r, S.size):
                z += 2 * S[jdx] ** 2 / (S[idx] ** 2 - S[jdx] ** 2)
        return z

    def SURE(r, S, v, Nx, Nt):
        return -2 * Nx * Nt * v + np.sum(S[r:] ** 2) + 4 * v * div_HARD(r, S, Nx, Nt)

    data_shape = data.shape[:3]
    M = np.full(data_shape, False)
    M[::step, ::step, ::step] = True
    ii, jj, kk = np.where(M)

    idx_i = np.arange(0, np.min((patch[0], data_shape[0])))
    idx_j = np.arange(0, np.min((patch[1], data_shape[1])))
    idx_k = np.arange(0, np.min((patch[2], data_shape[2])))

    navg = np.zeros((len(ii),) + data_shape)
    out = np.zeros_like(data)
    for count, (idx, jdx, kdx) in enumerate(zip(ii, jj, kk)):
        i = np.mod(idx + idx_i - 1, data_shape[0])
        j = np.mod(jdx + idx_j - 1, data_shape[1])
        k = np.mod(kdx + idx_k - 1, data_shape[2])

        curr_data = data[np.ix_(i, j, k)].reshape(-1, data[np.ix_(i, j, k)].shape[-1])

        U, S, V = lsvd(curr_data)
        s = np.diag(S)

        sure = np.zeros((curr_data.shape[0],))
        rank = np.arange(1, curr_data.shape[0] + 1)

        for r in rank:
            sure[r - 1] = SURE(r, s[::-1], var, curr_data.shape[0], curr_data.shape[-1])

        thresh_SURE = rank[np.argmin(sure)]
        # print(f'SURE hard = {thresh_SURE}')
        # breakpoint()

        S = S[-thresh_SURE:, -thresh_SURE:]
        V = V[:, -thresh_SURE:]
        U = U[:, -thresh_SURE:]

        out[np.ix_(i, j, k)] += (U @ S @ V.conj().T).reshape(
            data[np.ix_(i, j, k)].shape
        )

        navg[np.ix_([count], i, j, k)] += 1

    navg_all = np.sum(navg, axis=0)
    # Deal with broadcasting comparing along trailing dimension
    out = (out.T / navg_all.T).T

    return out


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


def save_data(clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray):
    """Save the data
    :param clean: the clean data
    :param noisy: the noisy data
    :param denoised: the denoised data
    :return: None
    """
    # save the data in the inference/results folder
    np.savez(
        cwd + "/inference_results_numpy_files/st_denoising/st_denoising_test_data.npz",
        clean_data=clean,
        noisy_data=noisy,
        reconstructed_data=denoised,
    )


if __name__ == "__main__":
    clean, noisy = load_data()

    # make a copy of the noisy data
    noisy_copy = noisy.copy()

    # calculate the noise standard deviation
    noise_std = np.std(clean[:, 0, :] - noisy[:, 0, :])

    # use j as the imaginary unit

    noisy = noisy[:, 0, :] + 1j * noisy[:, 1, :]

    # convert the noisy data to time domain
    noisy = np.fft.ifft(noisy, axis=0)

    # reshape to (10,1,1,512)
    noisy = noisy.reshape(200, 1, 1, -1)

    denoised_data = st_denoising(
        noisy, rank="mp", mp_var=2 * noise_std**2, patch=[3, 3, 1]
    )[0]

    # reshape the denoised data to (10,512)
    denoised_data = denoised_data.reshape(200, -1)
    noisy = noisy.reshape(200, -1)

    # convert the denoised data to frequency domain
    denoised_data = np.fft.fft(denoised_data, axis=0)

    # reshape denoised data to (200,1,512)
    denoised_data = denoised_data.reshape(200, 1, -1)

    # create array of 0s with shape as denoised_data
    array = np.zeros_like(denoised_data)

    # concatentae the array to the denoised_data along the axis 1 to get the shape (200, 2, 512)
    denoised_data = np.concatenate((denoised_data, array), axis=1)

    assert denoised_data.shape == noisy_copy.shape == clean.shape

    # save the denoised data
    save_data(clean, noisy_copy, denoised_data)
