# Hack to avoid circular imports
from pygelib.SO3_base import SO3Base


class SO3VecArray(SO3Base):
    """
    Core class for creating and tracking SO3 Vectors (aka SO3 representations).


    Parameters
    ----------

    data : iterable of of `torch.Tensor` with appropriate shape
        Input of a SO(3) vector.
    """

    @property
    def rdim(self):
        return -2

    @property
    def ells(self):
        return [(shape[self.rdim] - 1)//2 for shape in self.shapes]

    def add(self, other):
        return self.__add__(other)

    def __add__(self, other):
        """
        Add element wise `torch.Tensors`
        """
        return so3_torch.add(self, other)

    __radd__ = __add__

    def sub(self, other):
        return self.__sub__(other)

    def __sub__(self, other):
        """
        Subtract element wise `torch.Tensors`
        """
        return so3_torch.sub(self, other)

    __rsub__ = __sub__

    def mul(self, other):
        return self.__sub__(other)

    def complex_mul(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
        """
        Add element wise `torch.Tensors`
        """
        if isinstance(other, SO3ScalarArray):
            return multiply_VecArray_w_ScalarArray(self, other)

    __rmul__ = __mul__

    def div(self, other):
        return self.__true_div__(other)

    def __truediv__(self, other):
        """
        Add element wise `torch.Tensors`
        """
        return so3_torch.div(self, other)


def multiply_VecArray_w_ScalarArray(vec_array, scalar_array):
    output_parts = []
    for vec_part, scalar_part in zip(vec_array, scalar_array):
        scalar_r = scalar_part[0]
        scalar_i = scalar_part[1]

        vec_r = vec_part[0]
        vec_i = vec_part[1]

        output_vecs.append(torch.stack([vec_r*scalar_r - part_i*scalar_i, vec_r*scalar_i + part_i*scalar_r], dim=zdim))
    return SO3VecArray(output_parts)

    # return torch.stack([part_r*scalar_r - part_i*scalar_i, part_r*scalar_i + part_i*scalar_r], dim=zdim)

