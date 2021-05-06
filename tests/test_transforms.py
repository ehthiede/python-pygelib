import torch
from pygelib.transforms import DistanceBasisFeaturization, radial_gaussian
from torch_geometric.data import Data
import pytest


class TestDistanceBasisFeaturization:
    @pytest.mark.parametrize('alpha', [0., 1., 2.])
    def test_gaussian(self, alpha):
        r0s = torch.tensor([1., 2.])
        transform = DistanceBasisFeaturization(radial_gaussian, r0s=r0s, alpha=alpha)
        pos = torch.tensor([[1., 0., 0.], [2., 0., 0.], [3., 0., 0.]])
        data = Data(pos=pos)

        transformed = transform(data)

        ref_mat = torch.tensor([[0., 1., 2.],
                                [1., 0., 1.],
                                [2., 1., 0.]])

        refs = [torch.exp(-alpha * (ref_mat - r0)**2).ravel() for r0 in r0s]
        refs = torch.stack(refs, dim=-1)

        edges = []
        for i in range(3):
            for j in range(3):
                edges.append([i, j])

        ref_edge_indices = torch.tensor(edges).T.float()
        assert(torch.allclose(ref_edge_indices, transformed['edge_index']))
        assert(torch.allclose(refs, transformed['edge_attr']))
