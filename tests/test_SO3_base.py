import torch
from pygelib.SO3_base import SO3Base


class TestSO3Base(object):

    def test_tensor_mult(self):
        a_parts = [torch.randn(2, 4, 5) for i in range(3)]
        b_parts = [torch.randn(2, 4, 5) for i in range(3)]

        self._base_test_mult(a_parts, b_parts)

    def test_scalar_mult(self):
        a_parts = [torch.randn(2, 4, 5) for i in range(3)]
        b_parts = [torch.randn(2) for i in range(3)]
        self._base_test_mult(a_parts, b_parts)

    def _base_test_mult(self, a_parts, b_parts):
        a = SO3Base(a_parts)
        b = SO3Base(b_parts)

        c = a * b
        for a_pt, b_pt, c_pt in zip(a_parts, b_parts, c._data):
            assert(torch.allclose(c_pt[0], a_pt[0] * b_pt[0] - a_pt[1] * b_pt[1]))
            assert(torch.allclose(c_pt[1], a_pt[0] * b_pt[1] + a_pt[1] * b_pt[0]))

    def test_tensor_add(self):
        a_parts = [torch.randn(2, 4, 5) for i in range(3)]
        b_parts = [torch.randn(2, 4, 5) for i in range(3)]
        self._base_test_add(a_parts, b_parts)

    def test_scalar_add(self):
        a_parts = [torch.randn(2, 4, 5) for i in range(3)]
        b_parts = [torch.randn(2) for i in range(3)]
        self._base_test_add(a_parts, b_parts)

    def _base_test_add(self, a_parts, b_parts):
        a = SO3Base(a_parts)
        b = SO3Base(b_parts)

        c = a + b
        for a_pt, b_pt, c_pt in zip(a_parts, b_parts, c._data):
            assert(torch.allclose(c_pt[0], a_pt[0] + b_pt[0]))
            assert(torch.allclose(c_pt[1], a_pt[1] + b_pt[1]))

    def test_get_real(self):
        a_parts = [torch.randn(2, 4, 5) for i in range(3)]
        real_class = SO3Base(a_parts).real

        for a_in, a_out in zip(a_parts, real_class):
            assert(torch.allclose(a_in[0], a_out))
