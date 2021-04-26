from pygelib.CGproduct import _raw_cg_product, _compute_output_shape
import pytest
from pygelib.utils import _convert_to_SO3part_view
from pygelib.SO3VecArray import SO3VecArray
import torch


class TestCGProduct():
    # @pytest.mark.parametrize('lAs', [(0, 1, 4), (2,)])
    # @pytest.mark.parametrize('lBs', [(3, 5), (0, 1)])
    @pytest.mark.parametrize('lAs', [(0, 1, 4)])
    @pytest.mark.parametrize('lBs', [(0, 1)])
    @pytest.mark.parametrize('nc_A', [4])
    @pytest.mark.parametrize('num_vecs', [2])
    # @pytest.mark.parametrize('device', [torch.device('cuda'), torch.device('cpu')])
    @pytest.mark.parametrize('device', [torch.device('cpu')])
    def test_cgproduct_backward_A(self, lAs, lBs, nc_A,
                                  num_vecs, device):
        A_tnsrs = tuple([torch.randn(2, 1, 2*l+1, nc_A, device=device, requires_grad=True) for l in lAs])
        B_tnsrs = tuple([torch.randn(2, 1, 2*l+1, 3, device=device) for l in lBs])

        A_tnsrs = [_convert_to_SO3part_view(a, -2) for a in A_tnsrs]
        B_tnsrs = [_convert_to_SO3part_view(b, -2) for b in B_tnsrs]
        all_tensors = A_tnsrs + B_tnsrs

        num_A = len(A_tnsrs)

        output_info = _compute_output_shape(A_tnsrs, B_tnsrs)

        product_out = _raw_cg_product.apply(num_A, output_info, 0, None, *all_tensors)

        mean_0 = torch.mean(product_out[0])
        for p in product_out[1:]:
            mean_0 += torch.mean(p)
        # print(mean_0)
        # print(type(mean_0))
        mean_0.backward()
        # print(A_tnsrs[0].grad)
