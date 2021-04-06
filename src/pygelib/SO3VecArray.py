import torch
from pygelib.SO3TensorArray import SO3TensorArray
from pygelib.utils import move_to_end, move_from_end
from pygelib.rotations import WignerD_list


class SO3VecArray(SO3TensorArray):
    """
    Core class for creating and tracking SO3 Vectors (aka SO3 representations).


    Parameters
    ----------

    data : iterable of of `torch.Tensor` with appropriate shape
        Input of a SO(3) vector.
    """

    def __init__(self, data, rdim=2):
        if isinstance(data, type(self)):
            data = data.data
        else:
            self._data = list(data)

    @property
    def rdim(self):
        return 2

    @property
    def ells(self):
        return [(shape[self.rdim] - 1)//2 for shape in self.shapes]

    def rotate(self, alpha, beta, gamma):
        dtype = self._data[0].dtype
        device = self._data[0].device

        Dlist = WignerD_list(alpha, beta, gamma, dtype, device)

        new_data = []
        for i, (part, Dmat) in enumerate(zip(self._data, Dlist)):
            new_part = move_to_end(part, self.rdim)
            new_part = torch.matmul(new_part, Dmat)
            new_data.append(move_from_end(new_part, self.rdim))
        self._data = new_data


def pos_to_rep(pos, conj=False):
    r"""
    Convert a tensor of cartesian position vectors to an l=1 spherical tensor.

    Parameters
    ----------
    pos : :class:`torch.Tensor`
        A set of input cartesian vectors. Can have arbitrary batch dimensions
         as long as the last dimension has length three, for x, y, z.
    conj : :class:`bool`, optional
        Return the complex conjugated representation.


    Returns
    -------
    psi1 : :class:`torch.Tensor`
        The input cartesian vectors converted to a l=1 spherical tensor.

    """
    pos_x, pos_y, pos_z = pos.unbind(-1)

    # Only the y coordinates get mapped to imaginary terms
    if conj:
        pos_m = torch.stack([pos_x, pos_y], -1)/sqrt(2.)
        pos_p = torch.stack([-pos_x, pos_y], -1)/sqrt(2.)
    else:
        pos_m = torch.stack([pos_x, -pos_y], -1)/sqrt(2.)
        pos_p = torch.stack([-pos_x, -pos_y], -1)/sqrt(2.)
    pos_0 = torch.stack([pos_z, torch.zeros_like(pos_z)], -1)

    psi1 = torch.stack([pos_m, pos_0, pos_p], dim=-2).unsqueeze(-3)

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
