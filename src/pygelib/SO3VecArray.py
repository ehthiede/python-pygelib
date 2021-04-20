import torch
import numpy as np
from math import sqrt
from numbers import Number
from pygelib.SO3TensorArray import SO3TensorArray
from pygelib.utils import move_to_end, move_from_end
from pygelib.rotations import WignerD_list
from pygelib.utils import _convert_to_SO3part_view
from copy import deepcopy
import pygelib_cpp as backend


class SO3VecArray(SO3TensorArray):
    """
    Core class for creating and tracking SO3 Vectors (aka SO3 representations).


    Parameters
    ----------

    data : iterable of of `torch.Tensor` with appropriate shape
        Input functions that transform according to irreps of the
    rdim : int, optional
        Dimension upon which rotations act. Default is second-to-last.
    """

    def __init__(self, data, rdim=-2):
        if isinstance(data, type(self)):
            self._data = data._data
        elif isinstance(data, torch.Tensor):
            print("recognizing as tensor!")
            self._data = [data]
        else:
            self._data = list(data)

        self._rdim = -2

    @property
    def rdim(self):
        return self._rdim

    @property
    def ells(self):
        ls = []
        for shape in self.shapes:
            rshape = shape[self.rdim]
            assert(rshape % 2 == 1), "Rotational dimension is not odd!"
            ls.append((shape[self.rdim] - 1)//2)
        return ls

    @property
    def channels(self):
        return [shape[-1] for shape in self.shapes]

    @property
    def shapes(self):
        return [d.shape for d in self._data]

    @property
    def adims(self):
        arr_shapes = []
        for shape in self.shapes:
            new_shape = list(deepcopy(shape))
            del new_shape[self._rdim]
            # Remove first (complex) and last (channel) dimensions
            arr_shapes.append(tuple(new_shape[1:-1]))
        return arr_shapes

    @property
    def fragment_dict(self):
        fragment_dict = {}
        for shape in self.shapes:
            l = (shape[self.rdim] - 1)//2
            fragment_dict[l] = shape[-1]
        return fragment_dict

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


# ~~~ Initialization Routines ~~~ #
def zero_VecArray(n, fragment_dict, device=None):
    """
    Initializes an SO3VecArray that is full of zeros.
    Datatype is currently float32, since that is all that is supported by
    CGlib.

    Parameters
    ----------
    n : int or tuple
        Initial dimension(s) of the vec array
    fragment_dict : dict
        Dict where each key is the ell value, and the value is the number of associated channels.
    dtype : torch datatype object or strict
        Datatype to create
    """
    return _init_filled_array(n, fragment_dict, torch.zeros, device)


def ones_VecArray(n, fragment_dict, device=None):
    """
    Initializes an SO3VecArray that is full of zeros.
    Datatype is currently float32, since that is all that is supported by
    CGlib.

    Parameters
    ----------
    n : int or tuple
        Initial dimension(s) of the vec array
    fragment_dict : dict
        Dict where each key is the ell value, and the value is the number of associated channels.
    dtype : torch datatype object or strict
        Datatype to create
    """
    return _init_filled_array(n, fragment_dict, torch.ones, device)


def _init_filled_array(n, fragment_dict, fill_fxn, device=None):
    # Handle
    if isinstance(n, Number):
        n = (n,)
    else:
        n = tuple(n)

    data = []
    ls = []
    for l, channels in fragment_dict:
        shape = n + (l, channels)
        data.append(fill_fxn(shape, device=device))
        ls.append(ls)
    return SO3VecArray(data)


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


# ~~~ Conversion routines to the internal SO3partArray representations ~~~ #

def _convert_to_GELib(x, padding_multiple=32):
    """
    Converts an SO3vecArray into a collection of GElib Tensors.
    If the original SO3vecArray requires padding, this will result in an explicit copy.
    """
    rdim = x.rdim
    SO3part_list = []
    for x_l in x:
        # Add one to account for complex dim
        cell_index = rdim % len(np.shape(x_l)) + 1
        x_l_view = _convert_to_SO3part_view(x_l, cell_index,
                                            padding_multiple=padding_multiple)
        # Split into real, imaginary parts
        x_lv_r = x_l_view[0]
        x_lv_i = x_l_view[1]
        so3pt = backend._internal_SO3partArray_from_Tensor(x_lv_r, x_lv_i)
        SO3part_list.append(so3pt)
    return SO3part_list
