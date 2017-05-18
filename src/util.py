import scipy.io as sio
import numpy as np


def from_mat_to_path(path_mat, path_npy):
    """
    Transform the data which is the form of .mat to .npy
    """

    data = sio.loadmat(path_mat)
    np.save(path_npy, data)
