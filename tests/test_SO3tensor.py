import torch
import pytest
import operator
from pygelib.SO3TensorArray import SO3TensorArray


class TestSO3TensorArrayMult(object):
    # Pointwise Multiplication tests
    def test_array_mult(self):
        a_parts = [torch.randn(2, 4, 5) for i in range(3)]
        b_parts = [torch.randn(2, 4, 5) for i in range(3)]

        self._base_test_mult(a_parts, b_parts)

    def test_single_array_mult(self):
        a_parts = [torch.randn(2, 4, 5) for i in range(3)]
        b_parts = [torch.randn(2) for i in range(3)]
        self._base_test_mult(a_parts, b_parts)

    def _base_test_mult(self, a_parts, b_parts):
        a = SO3TensorArray(a_parts)
        b = SO3TensorArray(b_parts)

        c = a * b
        for a_pt, b_pt, c_pt in zip(a_parts, b_parts, c._data):
            assert(torch.allclose(c_pt[0], a_pt[0] * b_pt[0] - a_pt[1] * b_pt[1]))
            assert(torch.allclose(c_pt[1], a_pt[0] * b_pt[1] + a_pt[1] * b_pt[0]))

    @pytest.mark.parametrize('b', [-1, 2.1+0.4j,
                                   torch.complex(torch.randn(4, 5),
                                                 torch.randn(4, 5))
                                   ])
    def test_scalar_mult(self, b):
        a_parts = [torch.randn(2, 4, 5) for i in range(3)]
        a = SO3TensorArray(a_parts)
        c = a * b
        for a_pt, c_pt in zip(a_parts, c._data):
            assert(torch.allclose(c_pt[0], a_pt[0] * b.real - a_pt[1] * b.imag))
            assert(torch.allclose(c_pt[1], a_pt[0] * b.imag + a_pt[1] * b.real))

    def test_scalar_mult_real(self):
        a_parts = [torch.randn(2, 4, 5) for i in range(3)]
        a = SO3TensorArray(a_parts)
        b = torch.randn(4, 5)
        c = a * b
        for a_pt, c_pt in zip(a_parts, c._data):
            assert(torch.allclose(c_pt[0], a_pt[0] * b))
            assert(torch.allclose(c_pt[1], torch.zeros_like(a_pt[0])))


class TestSO3TensorArrayDiv(object):
    # Pointwise Multiplication tests
    def test_array_mult(self):
        a_parts = [torch.randn(2, 4, 5) for i in range(3)]
        b_parts = [torch.randn(2, 4, 5) for i in range(3)]

        self._base_test_mult(a_parts, b_parts)

    def test_single_array_mult(self):
        a_parts = [torch.randn(2, 4, 5) for i in range(3)]
        b_parts = [torch.randn(2) for i in range(3)]
        self._base_test_mult(a_parts, b_parts)

    def _base_test_mult(self, a_parts, b_parts):
        a = SO3TensorArray(a_parts)
        b = SO3TensorArray(b_parts)

        c = a / b
        for a_pt, b_pt, c_pt in zip(a_parts, b_parts, c._data):
            ref_real = (a_pt[0] * b_pt[0] + a_pt[1] * b_pt[1])
            ref_real /= (b_pt[0] * b_pt[0] + b_pt[1] * b_pt[1])
            ref_imag = (a_pt[1] * b_pt[0] - a_pt[0] * b_pt[1])
            ref_imag /= (b_pt[0] * b_pt[0] + b_pt[1] * b_pt[1])
            assert(torch.allclose(c_pt[0], ref_real))
            assert(torch.allclose(c_pt[1], ref_imag))

    @pytest.mark.parametrize('b', [-1, 2.1+0.4j,
                                   torch.complex(torch.randn(4, 5),
                                                 torch.randn(4, 5))
                                   ])
    def test_scalar_mult(self, b):
        a_parts = [torch.randn(2, 4, 5) for i in range(3)]
        a = SO3TensorArray(a_parts)
        c = a * b
        for a_pt, c_pt in zip(a_parts, c._data):
            assert(torch.allclose(c_pt[0], a_pt[0] * b.real - a_pt[1] * b.imag))
            assert(torch.allclose(c_pt[1], a_pt[0] * b.imag + a_pt[1] * b.real))

    def test_scalar_mult_real(self):
        a_parts = [torch.randn(2, 4, 5) for i in range(3)]
        a = SO3TensorArray(a_parts)
        b = torch.randn(4, 5)
        c = a * b
        for a_pt, c_pt in zip(a_parts, c._data):
            assert(torch.allclose(c_pt[0], a_pt[0] * b))
            assert(torch.allclose(c_pt[1], torch.zeros_like(a_pt[0])))


class TestSO3TensorArrayMatMul(object):
    # Matrix Multiplication tests
    def test_array_matmul(self):
        a_parts = [torch.randn(2, 4, 5) for i in range(3)]
        b_parts = [torch.randn(2, 5, 6) for i in range(3)]
        a = SO3TensorArray(a_parts)
        b = SO3TensorArray(b_parts)
        c = a @ b
        for a_pt, b_pt, c_pt in zip(a_parts, b_parts, c._data):
            assert(torch.allclose(c_pt[0], a_pt[0] @ b_pt[0] - a_pt[1] @ b_pt[1]))
            assert(torch.allclose(c_pt[1], a_pt[0] @ b_pt[1] + a_pt[1] @ b_pt[0]))

    def test_scalar_matmul(self):
        a_parts = [torch.randn(2, 4, 5) for i in range(3)]
        b = torch.complex(torch.randn(5, 3), torch.randn(5, 3))
        a = SO3TensorArray(a_parts)
        c = a @ b
        for a_pt, c_pt in zip(a_parts, c._data):
            assert(torch.allclose(c_pt[0], a_pt[0] @ b.real - a_pt[1] @ b.imag))
            assert(torch.allclose(c_pt[1], a_pt[0] @ b.imag + a_pt[1] @ b.real))

    def test_scalar_matmul_real(self):
        a_parts = [torch.randn(2, 4, 5) for i in range(3)]
        a = SO3TensorArray(a_parts)
        b = torch.randn(5, 3)
        c = a @ b
        for a_pt, c_pt in zip(a_parts, c._data):
            assert(torch.allclose(c_pt[0], a_pt[0] @ b))
            assert(torch.allclose(c_pt[1], torch.zeros_like(a_pt[0] @ b)))


class TestSO3TensorArrayAddSubtract(object):
    # Addition/Subtraction tests
    @pytest.mark.parametrize('op', [operator.add, operator.sub])
    def test_array_add(self, op):
        a_parts = [torch.randn(2, 4, 5) for i in range(3)]
        b_parts = [torch.randn(2, 4, 5) for i in range(3)]
        self._base_test_add(a_parts, b_parts, op)

    @pytest.mark.parametrize('op', [operator.add, operator.sub])
    def test_single_array_add(self, op):
        a_parts = [torch.randn(2, 4, 5) for i in range(3)]
        b_parts = [torch.randn(2) for i in range(3)]
        self._base_test_add(a_parts, b_parts, op)

    def _base_test_add(self, a_parts, b_parts, op):
        a = SO3TensorArray(a_parts)
        b = SO3TensorArray(b_parts)

        c = op(a, b)
        for a_pt, b_pt, c_pt in zip(a_parts, b_parts, c._data):
            assert(torch.allclose(c_pt[0], op(a_pt[0], b_pt[0])))
            assert(torch.allclose(c_pt[1], op(a_pt[1], b_pt[1])))

    @pytest.mark.parametrize('op', [operator.add, operator.sub])
    def test_tensor_add(self, op):
        a_parts = [torch.randn(2, 4, 5) for i in range(3)]
        a = SO3TensorArray(a_parts)
        b = torch.complex(torch.randn(4, 5), torch.randn(4, 5))
        c = op(a, b)
        for a_pt, c_pt in zip(a_parts, c._data):
            assert(torch.allclose(c_pt[0], op(a_pt[0], b.real)))
            assert(torch.allclose(c_pt[1], op(a_pt[1], b.imag)))

    @pytest.mark.parametrize('b', [-1, 2.1+0.4j,
                                   torch.complex(torch.randn(4, 5),
                                                 torch.randn(4, 5))
                                   ])
    @pytest.mark.parametrize('op', [operator.add, operator.sub])
    def test_scalar_add(self, b, op):
        a_parts = [torch.randn(2, 4, 5) for i in range(3)]
        a = SO3TensorArray(a_parts)
        c = op(a, b)
        for a_pt, c_pt in zip(a_parts, c._data):
            assert(torch.allclose(c_pt[0], op(a_pt[0], b.real)))
            assert(torch.allclose(c_pt[1], op(a_pt[1], b.imag)))

    @pytest.mark.parametrize('op', [operator.add, operator.sub])
    def test_scalar_add_real(self, op):
        a_parts = [torch.randn(2, 4, 5) for i in range(3)]
        a = SO3TensorArray(a_parts)
        b = torch.randn(4, 5)
        c = op(a, b)
        for a_pt, c_pt in zip(a_parts, c._data):
            assert(torch.allclose(c_pt[0], op(a_pt[0], b)))
            assert(torch.allclose(c_pt[1], a_pt[1]))


class TestSO3TensorArrayInterface(object):
    def test_get_real_imag(self):
        a_parts = [torch.randn(2, 4, 5) for i in range(3)]
        real_class = SO3TensorArray(a_parts).real
        imag_class = SO3TensorArray(a_parts).imag

        for a_in, a_out in zip(a_parts, real_class):
            assert(torch.allclose(a_in[0], a_out))

        for a_in, a_out in zip(a_parts, imag_class):
            assert(torch.allclose(a_in[1], a_out))
