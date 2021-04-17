import pytest
import torch
import pygelib_cpp as backend


class Test_SO3partArrayConversions():
    # @pytest.mark.parametrize('shapes', [((3, 5, 32), 1),
    #                                     ((4, 3, 64), 1),
    #                                     ((2, 2, 1, 32,), 2)])
    @pytest.mark.parametrize('shapes', [((3, 5, 32), 1)])
    # @pytest.mark.parametrize('device', [torch.device('cuda'), torch.device('cpu')])
    @pytest.mark.parametrize('device', [torch.device('cpu')])
    def test_conversions(self, shapes, device):
        shape, cell_index = shapes
        x_real = torch.randn(shape, device=device)
        x_imag = torch.randn(shape, device=device)
        y_real = torch.randn(shape, device=device)
        y_imag = torch.randn(shape, device=device)
        out_real = x_real + y_real
        out_imag = x_imag + y_imag

        x_SO3partArray = backend._internal_SO3partArray_from_Tensor(x_real, x_imag)
        y_SO3partArray = backend._internal_SO3partArray_from_Tensor(y_real, y_imag)
        backend.sum_SO3partArrays_inplace(x_SO3partArray, y_SO3partArray)
        assert(torch.allclose(x_imag, out_imag))
        assert(torch.allclose(x_real, out_real))
