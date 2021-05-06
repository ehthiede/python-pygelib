import torch
from math import sqrt


def pos_to_rep(pos, xyzdim=-2, conj=False):
    r"""
    Convert a tensor of cartesian position vectors to an l=1 spherical tensor.

    Parameters
    ----------
    pos : :class:`torch.Tensor`
        A set of input cartesian vectors. Can have arbitrary batch dimensions
        xyzdim dimension has length three, for x, y, z.

    conj : :class:`bool`, optional
        Return the complex conjugated representation.


    Returns
    -------
    psi1 : :class:`torch.Tensor`
        The input cartesian vectors converted to a l=1 spherical tensor.

    """
    pos_x, pos_y, pos_z = pos.unbind(xyzdim)

    # Only the y coordinates get mapped to imaginary terms
    if conj:
        pos_m = torch.stack([pos_x, pos_y], 0)/sqrt(2.)
        pos_p = torch.stack([-pos_x, pos_y], 0)/sqrt(2.)
    else:
        pos_m = torch.stack([pos_x, -pos_y], 0)/sqrt(2.)
        pos_p = torch.stack([-pos_x, -pos_y], 0)/sqrt(2.)
    pos_0 = torch.stack([pos_z, torch.zeros_like(pos_z)], 0)

    psi1 = torch.stack([pos_m, pos_0, pos_p], dim=xyzdim)

    return psi1


def rep_to_pos(rep):
    r"""
    Convert a tensor of l=1 spherical tensors to cartesian position vectors.

    Warning
    -------
    The input spherical tensor must satisfy :math:`F_{-m} = (-1)^m F_{m}^*`,
    so the output cartesian tensor is explicitly real. If this is not satisfied
    an error will be thrown.

    Parameters
    ----------
    rep : :class:`torch.Tensor`
        A set of input l=1 spherical tensors.
        Can have arbitrary batch dimensions as long
        as the last dimension has length three, for m = -1, 0, +1.

    Returns
    -------
    pos : :class:`torch.Tensor`
        The input l=1 spherical tensors converted to cartesian vectors.

    """
    rep_m, rep_0, rep_p = rep.unbind(-2)

    pos_x = (-rep_p + rep_m)/sqrt(2.)
    pos_y = (-rep_p - rep_m)/sqrt(2.)
    pos_z = rep_0

    imag_part = [pos_x[..., 1].abs().mean(), pos_y[..., 0].abs().mean(), pos_z[..., 1].abs().mean()]
    if (any(p > 1e-6 for p in imag_part)):
        raise ValueError('Imaginary part not zero! {}'.format(imag_part))

    pos = torch.stack([pos_x[..., 0], pos_y[..., 1], pos_z[..., 0]], dim=-1)

    return pos
