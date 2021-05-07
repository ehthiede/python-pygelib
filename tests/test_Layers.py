from pygelib.CG_routines import _raw_cg_product, _compute_output_shape
from pygelib.Layers import CGProduct, Linear, L1DifferenceLayer, ManyEdgeMPLayer
from pygelib.utils import _convert_to_SO3part_view
from pygelib.SO3VecArray import SO3VecArray
from pygelib.transforms import radial_gaussian
from pygelib.rotations import EulerRot
import pytest
import numpy as np
import torch


def _calculate_mean(tensor_list):
    mean_0 = torch.mean(tensor_list[0])
    for p in tensor_list[1:]:
        mean_0 += torch.mean(p)
    return mean_0


class TestCGProduct():
    @pytest.mark.parametrize('lAs', [(0, 1), (2,)])
    @pytest.mark.parametrize('lBs', [(0, 1, 3), (1, 4)])
    @pytest.mark.parametrize('nc_A', [1, 4])
    @pytest.mark.parametrize('num_vecs', [1, 2])
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


class TestLinear():
    @pytest.mark.parametrize('device', [torch.device('cpu'), torch.device('cuda')])
    @pytest.mark.parametrize('nc', [1, 4])
    @pytest.mark.parametrize('nc_out', [1, 4])
    @pytest.mark.parametrize('num_vecs', [1, 2])
    @pytest.mark.parametrize('ells', [(0, 1), (2,), (1, 3, 4)])
    def test_equivariance(self, ells, num_vecs, nc, nc_out, device):
        X = [torch.randn(2, num_vecs, 2*l+1, nc, device=device) for l in ells]

        X_rot = [torch.clone(a) for a in X]

        X = SO3VecArray(X)
        X_rot = SO3VecArray(X_rot)

        alpha, beta, gamma = tuple(np.random.randn(3))
        X_rot.rotate(alpha, beta, gamma)

        l_dict = {l: (nc, nc_out) for l in ells}
        lin = Linear(l_dict)
        lin.to(device)

        X_out_rot = lin(X)
        X_out_rot.rotate(alpha, beta, gamma)

        X_rot_out = lin(X_rot)

        for x, y in zip(X_out_rot, X_rot_out):
            print(torch.max(torch.abs(x - y)))
            assert(torch.allclose(x, y, atol=1e-6))


class TestL1DifferenceLayer():
    @pytest.mark.parametrize('device', [torch.device('cpu'), torch.device('cuda')])
    def test_equivariance(self, device):
        pos = torch.randn(9, 3)
        edge_idx = torch.randint(0, 5, size=(2, 20))
        edge_idx = torch.cat([edge_idx, torch.randint(5, 9, size=(2, 20))],
                             dim=1)

        node_features = torch.randn(9, 5)

        alpha, beta, gamma = tuple(np.random.randn(3))
        pos_rot = pos @ EulerRot(alpha, beta, gamma)

        r0s = torch.tensor([1., 2.])
        layer = L1DifferenceLayer(radial_gaussian, r0s=r0s, alpha=0.0)

        rep_out_rot = layer(pos, node_features, edge_idx)
        rep_out_rot.rotate(alpha, beta, gamma)
        rep_rot_out = layer(pos_rot, node_features, edge_idx)

        assert(torch.allclose(rep_out_rot[0], rep_rot_out[0], atol=5e-6))


class TestManyEdgeMPLayer():
    @pytest.mark.parametrize('ells', [(0, 1), (2,), (1, 3, 4)])
    @pytest.mark.parametrize('N', [4, 10])
    @pytest.mark.parametrize('only_real', [True, False])
    @pytest.mark.parametrize('device', [torch.device('cpu'), torch.device('cuda')])
    def test_equivariance(self, ells, N, only_real, device):
        nc = 5
        X = [torch.randn(2, N, 2*l+1, nc, device=device) for l in ells]
        X = SO3VecArray(X)
        X_rot = SO3VecArray([torch.clone(a) for a in X])
        alpha, beta, gamma = tuple(np.random.randn(3))

        X_rot.rotate(alpha, beta, gamma)
        X_rot = SO3VecArray(X_rot)

        layer = ManyEdgeMPLayer(only_real)

        # Build entries
        edge_indices = []
        for i in range(N):
            for j in range(N):
                if np.random.rand() > 0.5:
                    edge_indices.append([i, j])
        edge_idx = torch.tensor(np.array(edge_indices).T).to(device)
        M = edge_idx.shape[1]

        if only_real:
            edge_vals = torch.randn((M, 3), device=device)
        else:
            edge_vals = torch.randn((2, M, 3), device=device).T

        y = layer(X, edge_vals, edge_idx)
        y.rotate(alpha, beta, gamma)

        y_rot = layer(X_rot, edge_vals, edge_idx)

        for u_i, v_i in zip(y, y_rot):
            assert(torch.allclose(u_i, v_i, atol=1e-6))
