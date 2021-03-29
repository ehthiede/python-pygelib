import torch
from abc import ABC
from numbers import Number


class SO3Base(ABC):
    """
    Base class for a collection of tensors, each of which is associated with a
    specific value of ell.
    """
    def __init__(self, data):
        if isinstance(data, type(self)):
            data = data.data
        else:
            self._data = list(data)

    def __len__(self):
        """
        Length of SO3Vec.
        """
        return len(self._data)

    @property
    def maxl(self):
        """
        Maximum ell of SO3 object.

        Returns
        -------
        int
        """
        return len(self._data) - 1

    def truncate(self, maxl):
        """
        Update the maximum ell by truncating parts of the
        :class:`SO3Tensor` if they correspond to weights greater than `maxl`.

        Parameters
        ----------
        maxl : :obj:`int`
            Maximum weight to truncate the representation to.

        Returns
        -------
        :class:`SO3Base` subclass
            Truncated :class:`SO3Base`
        """
        return self[:maxl+1]

    @property
    def shapes(self):
        """
        Get a list of shapes of each :obj:`torch.Tensor` in the object.
        """
        return [p.shape for p in self]

    @property
    def device(self):
        if any(self._data[0].device != part.device for part in self._data):
            raise ValueError('Not all parts on same device!')

        return self._data[0].device

    @property
    def dtype(self):
        if any(self._data[0].dtype != part.dtype for part in self._data):
            raise ValueError('Not all parts using same data type!')

        return self._data[0].dtype

    @property
    def real(self):
        output_class = type(self)
        return output_class([i[0] for i in self._data])

    @property
    def imag(self):
        output_class = type(self)
        return output_class([i[0] for i in self._data])

    def keys(self):
        return range(len(self))

    def values(self):
        return iter(self._data)

    def items(self):
        return zip(range(len(self)), self._data)

    def __iter__(self):
        """
        Loop over contained :obj:`torch.tensor` objects
        """
        for t in self._data:
            yield t

    def __getitem__(self, idx):
        """
        Get a specific contained :obj:`torch.tensor` objects
        """
        if type(idx) is slice:
            return self.__class__(self._data[idx])
        else:
            return self._data[idx]

    def __setitem__(self, idx, val):
        """
        Set a specific contained :obj:`torch.tensor` objects
        """
        self._data[idx] = val

    def __eq__(self, other):
        """
        Check equality of two objects.
        """
        if len(self) != len(other):
            return False
        return all((part1 == part2).all() for part1, part2 in zip(self, other))

    @staticmethod
    def allclose(rep1, rep2, **kwargs):
        if len(rep1) != len(rep2):
            raise ValueError('')
        return all(torch.allclose(part1, part2, **kwargs) for part1, part2 in zip(rep1, rep2))

    def __str__(self):
        return str(list(self._data))

    __datar__ = __str__

    @classmethod
    def requires_grad(cls):
        # WHAT IS THIS AND WHY IS IT HERE?
        return cls([t.requires_grad() for t in self._data])

    def requires_grad_(self, requires_grad=True):
        self._data = [t.requires_grad_(requires_grad) for t in self._data]
        return self

    def to(self, *args, **kwargs):
        self._data = [t.to(*args, **kwargs) for t in self._data]
        return self

    def cpu(self):
        self._data = [t.cpu() for t in self._data]
        return self

    def cuda(self, **kwargs):
        self._data = [t.cuda(**kwargs) for t in self._data]
        return self

    def long(self):
        self._data = [t.long() for t in self._data]
        return self

    def byte(self):
        self._data = [t.byte() for t in self._data]
        return self

    def bool(self):
        self._data = [t.bool() for t in self._data]
        return self

    def half(self):
        self._data = [t.half() for t in self._data]
        return self

    def float(self):
        self._data = [t.float() for t in self._data]
        return self

    def double(self):
        self._data = [t.double() for t in self._data]
        return self

    def clone(self):
        return type(self)([t.clone() for t in self])

    def detach(self):
        return type(self)([t.detach() for t in self])

    @property
    def data(self):
        return self._data

    @property
    def grad(self):
        return type(self)([t.grad for t in self])

    def __mul__(self, other):
        """
        Add element wise `torch.Tensors`
        """
        if isinstance(other, SO3Base):
            return multiply_SO3Class_w_ScalarArray(self, other)
        else:
            raise Exception("Was unable to parse multiplied object.")

    # __rmul__ = __mul__

    def __add__(self, other):
        """
        Add element wise `torch.Tensors`
        """
        if isinstance(other, SO3Base):
            return add_SO3Class_w_ScalarArray(self, other)
        else:
            raise Exception("Was unable to parse multiplied object.")

    def __sub__(self, other):
        """
        Add element wise `torch.Tensors`
        """
        if isinstance(other, SO3Base):
            return add_SO3Class_w_ScalarArray(self, other)
        else:
            raise Exception("Was unable to parse multiplied object.")


def multiply_SO3Class_w_ScalarArray(vec_array, scalar_array):
    output_class = type(vec_array)
    output_parts = []
    for vec_part, scalar_part in zip(vec_array, scalar_array):
        scalar_r = scalar_part[0]
        scalar_i = scalar_part[1]

        vec_r = vec_part[0]
        vec_i = vec_part[1]

        output_parts.append(torch.stack([vec_r*scalar_r - vec_i*scalar_i, vec_r*scalar_i + vec_i*scalar_r], dim=0))
    return output_class(output_parts)


def add_SO3Class_w_ScalarArray(vec_array, scalar_array):
    output_class = type(vec_array)
    output_parts = []
    for vec_part, scalar_part in zip(vec_array, scalar_array):
        scalar_r = scalar_part[0]
        scalar_i = scalar_part[1]

        vec_r = vec_part[0]
        vec_i = vec_part[1]

        output_parts.append(torch.stack([vec_r+scalar_r, vec_i+scalar_i], dim=0))
    return output_class(output_parts)

# def multiply_SO3Class_w_Scalar(vec_array, scalar, scalar_is_complex=True):
#     output_class = type(vec_array)
#     output_parts = []

#     if scalar_is_complex:
#         scalar_r = scalar[0]
#         scalar_i = scalar[1]
#     else:
#         scalar_r = scalar
#         scalar_i = 0

#     for vec_part, scalar_part in zip(vec_array, scalar):
#         vec_r = vec_part[0]
#         vec_i = vec_part[1]

#         real_part = vec_r*scalar_r
#         if scalar_is_complex:
#             real_part -= vec_i*scalar_i

#         imag_part = vec_i*scalar_r
#         if scalar_is_complex:
#             imag_part += vec_r*scalar_i

#         output_parts.append(torch.stack([real_part, imag_part], dim=0))
#     return output_class(output_parts)
