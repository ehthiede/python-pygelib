"""
Test helper utilities for the cg product.
"""
import pytest
import torch
import numpy as np
import pygelib_cpp as backend
from pygelib.SO3VecArray import SO3VecArray
from pygelib.CGproduct import cg_product_forward, _compute_output_shape


class TestCGProduct():

    @pytest.mark.parametrize('lAs', [(0, 1, 4), (2,)])
    @pytest.mark.parametrize('lBs', [(3, 5), (0, 1)])
    @pytest.mark.parametrize('nc_A', [2, 4, 8])
    @pytest.mark.parametrize('nc_B', [3, 6])
    @pytest.mark.parametrize('num_vecs', [1, 3])
    # @pytest.mark.parametrize('device', [torch.device('cuda'), torch.device('cpu')])
    @pytest.mark.parametrize('device', [torch.device('cpu')])
    def test_cgproduct_reproducibility(self, lAs, lBs, nc_A,
                                       nc_B, num_vecs, device):
        A_tnsrs = [torch.randn(2, num_vecs, 2*l+1, nc_A, device=device) for l in lAs]
        B_tnsrs = [torch.randn(2, num_vecs, 2*l+1, nc_B, device=device) for l in lBs]

        A_tnsrs_copy = [torch.clone(a) for a in A_tnsrs]
        B_tnsrs_copy = [torch.clone(b) for b in B_tnsrs]

        A_arr = SO3VecArray(A_tnsrs)
        B_arr = SO3VecArray(B_tnsrs)

        A_arr_copy = SO3VecArray(A_tnsrs_copy)
        B_arr_copy = SO3VecArray(B_tnsrs_copy)

        C_out = cg_product_forward(A_arr, B_arr)

        C_out_copy = cg_product_forward(A_arr_copy, B_arr_copy)

        for i, j in zip(C_out, C_out_copy):
            torch.allclose(i - j, torch.zeros_like(i))


@pytest.mark.parametrize('l1_1', [1, 5])
@pytest.mark.parametrize('l1_2', [0, 4])
@pytest.mark.parametrize('l2_1', [1, 5])
@pytest.mark.parametrize('l2_2', [0, 4])
def test_compute_output_shape(l1_1, l1_2, l2_1, l2_2):
    """
    Explicitly checks predicted output shape,
    """
    # Perform the calculation in the frontend
    a_part_1 = torch.randn(2, 1, 2 * l1_1 + 1, 8)
    a_part_2 = torch.randn(2, 1, 2 * l1_2 + 1, 16)
    a = SO3VecArray([a_part_1, a_part_2])

    b_part_1 = torch.randn(2, 1, 2 * l2_1 + 1, 4)
    b_part_2 = torch.randn(2, 1, 2 * l2_2 + 1, 8)
    b = SO3VecArray([b_part_1, b_part_2])
    output_keys, predicted_shapes = _compute_output_shape(a, b)

    # Perform the calculation in GElib
    gaussian_fill = backend._fill_gaussian()
    a_gelib_1 = backend._SO3partArray([1], l1_1, 8, gaussian_fill, 0)
    a_gelib_2 = backend._SO3partArray([1], l1_2, 16, gaussian_fill, 0)
    a_gelibs = [a_gelib_1, a_gelib_2]
    b_gelib_1 = backend._SO3partArray([1], l2_1, 4, gaussian_fill, 0)
    b_gelib_2 = backend._SO3partArray([1], l2_2, 8, gaussian_fill, 0)
    b_gelibs = [b_gelib_1, b_gelib_2]

    l1max = max(l1_1, l1_2)
    l2max = max(l2_1, l2_2)

    for l_out in range(0, l1max + l2max):
        gelib_product_parts = []
        # Calculate products over all pairs of parts
        for atns in a_gelibs:
            for btns in b_gelibs:
                prod = backend.partArrayCGproduct(atns, btns, l_out)
                gelib_product_parts.append(prod)

        nonzero_part_channels = []
        for part in gelib_product_parts:
            product_tensors = backend._internal_Tensor_from_SO3partArray(part)
            product_tensor = torch.stack(product_tensors, dim=0)
            if not torch.allclose(product_tensor, torch.zeros_like(product_tensor)):
                nonzero_part_channels.append(backend.get_num_channels(part))

        if len(nonzero_part_channels) > 0:
            num_channels = np.sum(nonzero_part_channels)
        else:
            num_channels = 0

        if l_out in predicted_shapes.keys():
            num_channels_predicted = predicted_shapes[l_out][-1]
            assert(num_channels == num_channels_predicted)
        else:
            assert(num_channels == 0)
