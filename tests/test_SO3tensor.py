import torch
from pygelib.SO3Tensor import SO3Tensor


class TestSO3Tensor(object):

    def test_array_mult(self):
        a_parts = [torch.randn(2, 4, 5) for i in range(3)]
        b_parts = [torch.randn(2, 4, 5) for i in range(3)]

        self._base_test_mult(a_parts, b_parts)

    def test_single_array_mult(self):
        a_parts = [torch.randn(2, 4, 5) for i in range(3)]
        b_parts = [torch.randn(2) for i in range(3)]
        self._base_test_mult(a_parts, b_parts)

    def _base_test_mult(self, a_parts, b_parts):
        a = SO3Tensor(a_parts)
        b = SO3Tensor(b_parts)

        c = a * b
        for a_pt, b_pt, c_pt in zip(a_parts, b_parts, c._data):
            assert(torch.allclose(c_pt[0], a_pt[0] * b_pt[0] - a_pt[1] * b_pt[1]))
            assert(torch.allclose(c_pt[1], a_pt[0] * b_pt[1] + a_pt[1] * b_pt[0]))

    def test_tensor_mult(self):
        a_parts = [torch.randn(2, 4, 5) for i in range(3)]
        a = SO3Tensor(a_parts)
        b = torch.complex(torch.randn(4, 5), torch.randn(4, 5))
        c = a * b
        for a_pt, c_pt in zip(a_parts, c._data):
            assert(torch.allclose(c_pt[0], a_pt[0] * b.real - a_pt[1] * b.imag))
            assert(torch.allclose(c_pt[1], a_pt[0] * b.imag + a_pt[1] * b.real))

    def test_array_add(self):
        a_parts = [torch.randn(2, 4, 5) for i in range(3)]
        b_parts = [torch.randn(2, 4, 5) for i in range(3)]
        self._base_test_add(a_parts, b_parts)

    def test_single_array_add(self):
        a_parts = [torch.randn(2, 4, 5) for i in range(3)]
        b_parts = [torch.randn(2) for i in range(3)]
        self._base_test_add(a_parts, b_parts)

    def _base_test_add(self, a_parts, b_parts):
        a = SO3Tensor(a_parts)
        b = SO3Tensor(b_parts)

        c = a + b
        for a_pt, b_pt, c_pt in zip(a_parts, b_parts, c._data):
            assert(torch.allclose(c_pt[0], a_pt[0] + b_pt[0]))
            assert(torch.allclose(c_pt[1], a_pt[1] + b_pt[1]))

    def test_tensor_add(self):
        a_parts = [torch.randn(2, 4, 5) for i in range(3)]
        a = SO3Tensor(a_parts)
        b = torch.complex(torch.randn(4, 5), torch.randn(4, 5))
        c = a + b
        for a_pt, c_pt in zip(a_parts, c._data):
            assert(torch.allclose(c_pt[0], a_pt[0] + b.real))
            assert(torch.allclose(c_pt[1], a_pt[1] + b.imag))


    def test_get_real(self):
        a_parts = [torch.randn(2, 4, 5) for i in range(3)]
        real_class = SO3Tensor(a_parts).real

        for a_in, a_out in zip(a_parts, real_class):
            assert(torch.allclose(a_in[0], a_out))
