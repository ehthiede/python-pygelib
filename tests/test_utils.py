import torch
import pytest
from pygelib.utils import _prune_self_loops


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
