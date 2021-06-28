import torch
import pytest
import numpy as np
from pygelib.models import basic_cg_model as bcm
from pygelib.SO3VecArray import SO3VecArray
from torch_geometric.data import Data
from pygelib.rotations import EulerRot


class TestBasicCGModel():
    @pytest.mark.parametrize('lmax', [2, 5])
    # @pytest.mark.parametrize('lmax', [2])
    @pytest.mark.parametrize('N', [8, 10])
    @pytest.mark.parametrize('nc', [2, 6])
    @pytest.mark.parametrize('n_layers', [2, 4, 5])
    # @pytest.mark.parametrize('device', [torch.device('cpu'), torch.device('cuda')])
    @pytest.mark.parametrize('device', [torch.device('cpu')])
    def test_rotational_equivariance(self, lmax, N, nc, n_layers, device):
        x = torch.randn(N, 5, device=device)
        r0s = torch.tensor([1., 2.], device=device)

        # Build entries of edges
        batch_indices = torch.zeros(N, dtype=torch.int64, device=device)
        batch_indices[5:] = 1

        edge_indices = [[i, i] for i in range(N)]
        for i in range(5):
            for j in range(5):
                if ((np.random.rand() > 0.5) and (i != j)):
                    edge_indices.append([i, j])
        for i in range(5, N):
            for j in range(5, N):
                if ((np.random.rand() > 0.5) and (i != j)):
                    edge_indices.append([i, j])

        print(edge_indices)

        edge_idx = torch.tensor(np.array(edge_indices).T).to(device)
        M = edge_idx.shape[1]
        edge_vals = torch.randn((M, 4), device=device)

        # Build positions and rotated positions
        pos = torch.randn(N, 3, device=device)
        alpha, beta, gamma = tuple(np.random.randn(3))
        pos_rot = pos @ EulerRot(alpha, beta, gamma).to(device)

        # Build data structure
        data = Data(x=x, pos=pos, edge_attr=edge_vals, edge_idx=edge_idx, batch=batch_indices)
        data_rot = Data(x=x, pos=pos_rot, edge_attr=edge_vals, edge_idx=edge_idx, batch=batch_indices)

        model = bcm.CGMPModel(5, 4, nc, n_layers, lmax, r0s).to(device=device)

        y = model(data)
        y_rot = model(data_rot)

        assert(torch.allclose(y, y_rot, atol=1e-3))


class TestBasicCGMPBlock():
    @pytest.mark.parametrize('ells', [(0, 1), (2,)])
    @pytest.mark.parametrize('N', [4, 10])
    @pytest.mark.parametrize('nc', [2, 4])
    @pytest.mark.parametrize('nc_out', [2, 4])
    @pytest.mark.parametrize('only_real', [True, False])
    @pytest.mark.parametrize('device', [torch.device('cpu'), torch.device('cuda')])
    def test_rotational_equivariance(self, ells, N, nc, nc_out, only_real, device):
        X = [torch.randn(2, N, 2*l+1, nc, device=device) for l in ells]

        X_rot = [torch.clone(a) for a in X]

        X = SO3VecArray(X)
        X_rot = SO3VecArray(X_rot)

        alpha, beta, gamma = tuple(np.random.randn(3))
        X_rot.rotate(alpha, beta, gamma)

        # Build entries of edges
        edge_indices = []
        for i in range(N):
            for j in range(N):
                if np.random.rand() > 0.5:
                    edge_indices.append([i, j])
        edge_idx = torch.tensor(np.array(edge_indices).T).to(device)
        M = edge_idx.shape[1]

        num_edge_channels = 3

        if only_real:
            edge_vals = torch.randn((M, num_edge_channels), device=device)
        else:
            edge_vals = torch.randn((2, M, num_edge_channels), device=device)

        l_dict = {}
        for l in range(3):
            if l in ells:
                l_dict[l] = (nc, nc_out)
            else:
                l_dict[l] = (0, nc_out)

        for l in range(3, 5):
            if l in ells:
                l_dict[l] = (nc, 0)

        block = bcm.CGMPBlock(l_dict, num_edge_channels, real_edges=only_real)
        block.to(device)

        X_out_rot = block(X, edge_vals, edge_idx)
        X_out_rot.rotate(alpha, beta, gamma)

        X_rot_out = block(X_rot, edge_vals, edge_idx)

        for x, y in zip(X_out_rot, X_rot_out):
            print(torch.max(torch.abs(x - y)))
            assert(torch.allclose(x, y, atol=5e-5))
