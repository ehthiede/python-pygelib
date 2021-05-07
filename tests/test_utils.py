import torch
import pytest
import numpy as np
from pygelib.utils import _prune_self_loops
from pygelib.utils import _complex_spmm


@pytest.mark.parametrize('use_pair_vals', [True, False])
def test_prune_self_loops(use_pair_vals):
    if use_pair_vals:
        vals = torch.eye(10).float().ravel()
    else:
        vals = None

    edges = []
    for i in range(10):
        for j in range(10):
            edges.append([i, j])

    edges = torch.tensor(edges).T

    pruned_indices, pruned_edge_vals = _prune_self_loops(edges, vals)

    indx_diff = pruned_indices[1] - pruned_indices[0]
    assert(all(indx_diff != 0))
    if use_pair_vals:
        assert(torch.allclose(pruned_edge_vals, torch.zeros_like(pruned_edge_vals)))


class TestComplexSPMM:
    @pytest.mark.parametrize('N', [4, 10])
    @pytest.mark.parametrize('device', [torch.device('cpu'), torch.device('cuda')])
    def test_real_sparse_mat(self, N, device):
        # Build dense numpy analogues
        edge_indices = []
        edge_vals = []
        for i in range(N):
            for j in range(N):
                if np.random.rand() > 0.5:
                    edge_indices.append([i, j])
                    edge_vals.append(np.random.randn())
        edge_idx_np = np.array(edge_indices).T
        edge_vals_np = np.array(edge_vals)

        numpy_mat = np.zeros((N, N))
        numpy_mat[edge_idx_np[0], edge_idx_np[1]] = edge_vals_np

        x = np.random.randn(2, N, 5)
        x_np = x[0] + 1j * x[1]

        y_np = numpy_mat @ x_np

        # Perform pytorch calculation
        edge_idx = torch.from_numpy(edge_idx_np).to(device)
        edge_vals = torch.from_numpy(edge_vals_np).to(device)

        x = torch.from_numpy(x).to(device)
        y = _complex_spmm(edge_idx, edge_vals, N, x, sparse_is_real=True)
        y = y.cpu().detach().numpy()
        y = y[0] + 1j * y[1]

        assert(np.allclose(y_np, y))

    @pytest.mark.parametrize('N', [4, 10])
    @pytest.mark.parametrize('device', [torch.device('cpu'), torch.device('cuda')])
    def test_imag_sparse_mat(self, N, device):
        # Build dense numpy analogues
        edge_indices = []
        edge_vals = []
        for i in range(N):
            for j in range(N):
                if np.random.rand() > 0.5:
                    edge_indices.append([i, j])
                    edge_vals.append(np.random.randn(2))
        edge_idx_np = np.array(edge_indices).T
        edge_vals_np = np.array(edge_vals).T

        numpy_mat = np.zeros((N, N), dtype=np.complex64)
        numpy_mat[edge_idx_np[0], edge_idx_np[1]] = edge_vals_np[0]
        numpy_mat[edge_idx_np[0], edge_idx_np[1]] += 1j * edge_vals_np[1]

        x = np.random.randn(2, N, 5)
        x_np = x[0] + 1j * x[1]

        y_np = numpy_mat @ x_np

        # Perform pytorch calculation
        edge_idx = torch.from_numpy(edge_idx_np).to(device)
        edge_vals = torch.from_numpy(edge_vals_np).to(device)

        x = torch.from_numpy(x).to(device)
        y = _complex_spmm(edge_idx, edge_vals, N, x, sparse_is_real=False)
        y = y.cpu().detach().numpy()
        y = y[0] + 1j * y[1]

        assert(np.allclose(y_np, y))
