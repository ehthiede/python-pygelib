import pytest
import torch
import pygelib_cpp as backend


class Test_SO3partArrayConversions():
    @pytest.mark.parametrize('shapes', [((3, 5, 32), 1),
                                        ((4, 3, 64), 1),
                                        ((2, 2, 1, 32,), 2)])
    @pytest.mark.parametrize('device', [torch.device('cuda'), torch.device('cpu')])
    def test_conversions(self, shapes, device):
        shape, cell_index = shapes
        x_real = torch.randn(shape, device=device)
        x_imag = torch.randn(shape, device=device)
        print(x_real.device)
        print("-----------")
        part_array = backend._construct_SO3partArray_from_Tensor(x_real, x_imag)
        y_real, y_imag = backend._construct_Tensor_from_SO3partArray(part_array)
        assert(torch.allclose(x_real, y_real))
        assert(torch.allclose(x_imag, y_imag))
