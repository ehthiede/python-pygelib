from pygelib.CGproduct import _raw_cg_product, _compute_output_shape, CGProduct
from pygelib.utils import _convert_to_SO3part_view
from pygelib.SO3VecArray import SO3VecArray
import pytest
import torch


def _calculate_mean(tensor_list):
    mean_0 = torch.mean(tensor_list[0])
    for p in tensor_list[1:]:
        mean_0 += torch.mean(p)
    return mean_0


class TestCGProduct():
    @pytest.mark.parametrize('lAs', [(0, 1), (2,)])
    @pytest.mark.parametrize('lBs', [(0, 1, 3), (1, 4)])
    # @pytest.mark.parametrize('lBs', [(1,)])
    @pytest.mark.parametrize('nc_A', [1, 4])
    @pytest.mark.parametrize('num_vecs', [1, 2])
    # @pytest.mark.parametrize('device', [torch.device('cuda')])
    @pytest.mark.parametrize('device', [torch.device('cpu')])
    def test_cgproduct_matches_raw(self, lAs, lBs, nc_A, num_vecs, device):
        A_tnsrs = [torch.randn(2, num_vecs, 2*l+1, nc_A, device=device) for l in lAs]
        B_tnsrs = [torch.randn(2, num_vecs, 2*l+1, 3, device=device) for l in lBs]

        num_A = len(A_tnsrs)

        A_tnsrs_copy = [torch.clone(a) for a in A_tnsrs]
        B_tnsrs_copy = [torch.clone(b) for b in B_tnsrs]

        # Make everything require grad.
        for tnsr_list in [A_tnsrs, B_tnsrs, A_tnsrs_copy, B_tnsrs_copy]:
            for t in tnsr_list:
                t.requires_grad = True

        A = SO3VecArray([_convert_to_SO3part_view(a, -2) for a in A_tnsrs])
        B = SO3VecArray([_convert_to_SO3part_view(b, -2) for b in B_tnsrs])

        # A, B get sent to a tensor list for the raw codes)
        raw_tensors = [_convert_to_SO3part_view(a, -2) for a in A_tnsrs_copy]
        raw_tensors += [_convert_to_SO3part_view(b, -2) for b in B_tnsrs_copy]

        # Initialize Class
        output_info = _compute_output_shape(A, B)
        product_instance = CGProduct(output_info, 0, None)

        class_out = product_instance(A, B)

        raw_out = _raw_cg_product.apply(num_A, output_info, 0, None, *raw_tensors)

        for i, j in zip(class_out, raw_out):
            assert(torch.allclose(i, j))

        # Calculate backwards passes
        _calculate_mean(class_out).backward()
        _calculate_mean(raw_out).backward()

        for a_i, a_j in zip(A_tnsrs, A_tnsrs_copy):
            assert(torch.allclose(a_i.grad, a_j.grad))

        for b_i, b_j in zip(B_tnsrs, B_tnsrs_copy):
            assert(torch.allclose(b_i.grad, b_j.grad))
