"""
Test helper utilities for the cg product.
"""
import pytest
import torch
import numpy as np
import pygelib_cpp as backend
from pygelib.SO3VecArray import SO3VecArray
from pygelib.CG_routines import _cg_product_forward, _compute_output_shape, _raw_cg_product
from pygelib.utils import _convert_to_SO3part_view


class TestCGProductBackend():
    @pytest.mark.parametrize('lAs', [(0, 1), (2,)])
    @pytest.mark.parametrize('lBs', [(0, 1, 3)])
    @pytest.mark.parametrize('num_vecs', [1, 2])
    # @pytest.mark.parametrize('device', [torch.device('cuda'), torch.device('cpu')])
    @pytest.mark.parametrize('device', [torch.device('cpu')])
    # @pytest.mark.parametrize('device', [torch.device('cuda')])
    def test_cgproduct_backward_A(self, lAs, lBs, num_vecs, device):
        A_tnsrs = tuple([torch.randn(2, num_vecs, 2*l+1, 2, device=device) for l in lAs])
        B_tnsrs = tuple([torch.randn(2, num_vecs, 2*l+1, 3, device=device) for l in lBs])

        A_tnsrs = [_convert_to_SO3part_view(a, -2) for a in A_tnsrs]
        B_tnsrs = [_convert_to_SO3part_view(b, -2) for b in B_tnsrs]

        for A in A_tnsrs:
            A.requires_grad = True

        tensors = A_tnsrs + B_tnsrs

        num_A = len(A_tnsrs)
        output_info = _compute_output_shape(A_tnsrs, B_tnsrs)

        def wrapped_cgproduct(*all_tensors):
            return _raw_cg_product.apply(num_A, output_info, 0, None, *all_tensors)

        # print([t for t in tensors])
        passed_test = torch.autograd.gradcheck(wrapped_cgproduct, tuple(tensors), eps=1e-3, atol=1e-3, rtol=1e-3)

        assert(passed_test)


    @pytest.mark.parametrize('lAs', [(0, 1, 4), (2,)])
    @pytest.mark.parametrize('lBs', [(3, 5), (0, 1)])
    @pytest.mark.parametrize('nc_A', [2, 4])
    @pytest.mark.parametrize('num_vecs', [1, 2])
    @pytest.mark.parametrize('device', [torch.device('cuda'), torch.device('cpu')])
    # @pytest.mark.parametrize('device', [torch.device('cpu')])
    def test_cgproduct_reproducibility(self, lAs, lBs, nc_A,
                                       num_vecs, device):

        A_tnsrs = [torch.randn(2, num_vecs, 2*l+1, nc_A, device=device) for l in lAs]
        B_tnsrs = [torch.randn(2, num_vecs, 2*l+1, 8, device=device) for l in lBs]
        A_tnsrs_copy = [torch.clone(a) for a in A_tnsrs]
        B_tnsrs_copy = [torch.clone(b) for b in B_tnsrs]

        A_tnsrs = [_convert_to_SO3part_view(a, -2) for a in A_tnsrs]
        B_tnsrs = [_convert_to_SO3part_view(b, -2) for b in B_tnsrs]

        A_tnsrs_copy = [_convert_to_SO3part_view(a, -2) for a in A_tnsrs_copy]
        B_tnsrs_copy = [_convert_to_SO3part_view(b, -2) for b in B_tnsrs_copy]

        A_arr = A_tnsrs
        B_arr = B_tnsrs

        A_arr_copy = A_tnsrs_copy
        B_arr_copy = B_tnsrs_copy

        C_out = _cg_product_forward(A_arr, B_arr)
        C_out_copy = _cg_product_forward(A_arr_copy, B_arr_copy)

        for i, j in zip(C_out, C_out_copy):
            assert(torch.allclose(i, j))
            is_abnormally_large = (torch.abs(i) > 1e9).float()
            assert(torch.allclose(is_abnormally_large, torch.zeros_like(i)))

    @pytest.mark.parametrize('lA', [0, 1, 2])
    @pytest.mark.parametrize('lB', [0, 1, 3])
    @pytest.mark.parametrize('device', [torch.device('cuda'), torch.device('cpu')])
    # @pytest.mark.parametrize('device', [torch.device('cpu')])
    def test_product_values(self, lA, lB, device):
        nc_A = 4
        nc_B = 8

        num_atoms = 3

        A_tnsr = torch.randn(2, num_atoms, 2*lA+1, nc_A, device=device)
        B_tnsr = torch.randn(2, num_atoms, 2*lB+1, nc_B, device=device)

        A_tnsr = _convert_to_SO3part_view(A_tnsr, -2)
        B_tnsr = _convert_to_SO3part_view(B_tnsr, -2)

        A_arr = [A_tnsr]
        B_arr = [B_tnsr]

        C_out_copy = _cg_product_forward(A_arr, B_arr)

        A_gelib_prt = backend._internal_SO3partArray_from_Tensor(A_tnsr[0], A_tnsr[1])
        B_gelib_prt = backend._internal_SO3partArray_from_Tensor(B_tnsr[0], B_tnsr[1])

        for c in C_out_copy:
            l = (c.shape[2] - 1) // 2
            c_flat = c.reshape(2, num_atoms, c.shape[2] * c.shape[3])

            c_gelib_prod = backend.partArrayCGproduct(A_gelib_prt, B_gelib_prt, l)
            c_from_gelib = backend._internal_Tensor_from_SO3partArray(c_gelib_prod)
            c_from_gelib = torch.stack(c_from_gelib, dim=0)

            assert(torch.allclose(c_flat, c_from_gelib))

    @pytest.mark.parametrize('lAs', [(0, 1, 3), (2,)])
    @pytest.mark.parametrize('lBs', [(2, 3), (0, 1)])
    @pytest.mark.parametrize('nc_A', [2, 4])
    @pytest.mark.parametrize('nc_B', [3, 8])
    @pytest.mark.parametrize('device', [torch.device('cuda'), torch.device('cpu')])
    def test_cgproduct_equivariance(self, lAs, lBs, nc_A,
                                    nc_B, device):

        # Setup the input tensors....
        A_tnsrs = [torch.randn(2, 1, 2*l+1, nc_A, device=device) for l in lAs]
        B_tnsrs = [torch.randn(2, 1, 2*l+1, nc_B, device=device) for l in lBs]

        A_tnsrs_rot = [torch.clone(a) for a in A_tnsrs]
        B_tnsrs_rot = [torch.clone(b) for b in B_tnsrs]

        # Initialize unrotated vectors
        A_tnsrs = [_convert_to_SO3part_view(a, -2) for a in A_tnsrs]
        B_tnsrs = [_convert_to_SO3part_view(b, -2) for b in B_tnsrs]
        A_vec = A_tnsrs
        B_vec = B_tnsrs

        # Initialize rotated vectors
        alpha, beta, gamma = tuple(np.random.randn(3))

        # Typecast to SO3VecArray for easy rotations
        A_vec_rot = SO3VecArray(A_tnsrs_rot)
        A_vec_rot.rotate(alpha, beta, gamma)
        B_vec_rot = SO3VecArray(B_tnsrs_rot)
        B_vec_rot.rotate(alpha, beta, gamma)
        A_vec_rot = SO3VecArray([_convert_to_SO3part_view(a, -2) for a in A_vec_rot])
        B_vec_rot = SO3VecArray([_convert_to_SO3part_view(b, -2) for b in B_vec_rot])

        CG_out_rot = SO3VecArray(_cg_product_forward(A_vec, B_vec))
        CG_out_rot.rotate(alpha, beta, gamma)
        CG_rot_out = _cg_product_forward(A_vec_rot, B_vec_rot)

        for i, j in zip(CG_out_rot, CG_rot_out):
            assert(torch.allclose(i, j, atol=1e-5))


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
    a = [a_part_1, a_part_2]

    b_part_1 = torch.randn(2, 1, 2 * l2_1 + 1, 4)
    b_part_2 = torch.randn(2, 1, 2 * l2_2 + 1, 8)
    b = [b_part_1, b_part_2]
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
