"""
Routines for constructing WignerD matrices

"""

import torch
import numpy as np


def create_Jy(j):
    mrange = -np.arange(-j, j)
    jp_diag = np.sqrt((j+mrange)*(j-mrange+1))
    Jp = np.diag(jp_diag, k=1)
    Jm = np.diag(jp_diag, k=-1)

    Jy = -(Jp - Jm) / complex(0, 2)

    return Jy


def littled(j, beta):
    Jy = create_Jy(j)

    evals, evecs = np.linalg.eigh(Jy)
    evecsh = evecs.conj().T
    evals_exp = np.diag(np.exp(complex(0, -beta)*evals))

    d = np.matmul(np.matmul(evecs, evals_exp), evecsh)

    return d


def WignerD(j, alpha, beta, gamma, dtype=torch.float, device=None):
    """
    Calculates the Wigner D matrix for a given degree and Euler Angle.

    Parameters
    ----------
    j : int
        Degree of the representation.
    alpha : double
        First Euler angle
    beta : double
        Second Euler angle
    gamma : double
        Third Euler angle
    device : :obj:`torch.device`, optional
        Device of the output tensor
    dtype : :obj:`torch.dtype`, optional
        Data dype of the output tensor

    Returns
    -------
    D =


    """
    if device is None:
        device = torch.device('cpu')
    d = littled(j, beta)

    Jz = np.arange(-j, j+1)
    Jzl = np.expand_dims(Jz, 1)

    # np.multiply() broadcasts, so this isn't actually matrix multiplication, and 'left'/'right' are lies
    left = np.exp(complex(0, -alpha)*Jzl)
    right = np.exp(complex(0, -gamma)*Jz)

    D = left * d * right

    D = complex_from_numpy(D, dtype=dtype, device=device)

    return D


def WignerD_list(jmax, alpha, beta, gamma, dtype=torch.float, device=None):
    """

    """
    if device is None:
        device = torch.device('cpu')
    return [WignerD(j, alpha, beta, gamma, dtype=dtype, device=device) for j in range(jmax+1)]


def complex_from_numpy(z, dtype=torch.float, device=None):
    """ Take a numpy array and output a complex array of the same size. """
    if device is None:
        device = torch.device('cpu')
    zr = torch.from_numpy(z.real).to(dtype=dtype, device=device)
    zi = torch.from_numpy(z.imag).to(dtype=dtype, device=device)

    return torch.stack((zr, zi), 0)


def _Ry(theta):
    """
    Rotation Matrix for rotations on the y axis.
    """
    return torch.tensor([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]], dtype=torch.double)


def _Rz(theta):
    """
    Rotation Matrix for rotations on the z axis. Syntax is the same as with Ry.
    """
    return torch.tensor([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]], dtype=torch.double)


def EulerRot(alpha, beta, gamma):
    """
    Constructs a Rotation Matrix from Euler angles.

    Parameters
    ----------
    alpha : double
        First Euler angle
    beta : double
        Second Euler angle
    gamma : double
        Third Euler angle

    Returns
    -------
    Rmat : :obj:`torch.Tensor`
        The associated rotation matrix.
    """
    return Rz(alpha) @ Ry(beta) @ Rz(gamma)

