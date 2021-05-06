import torch
from torch_geometric.data import Data
from pygelib.utils import _prune_self_loops


class Atom3DPositionTransform:
    def __init__(self, atom_labels, label_idx=None):
        self.atom_labels = atom_labels
        self.label_idx = None

    def __call__(self, data):
        pos = torch.from_numpy(data['atoms'][['x', 'y', 'z']].to_numpy())
        elements = data['atoms'].element.to_numpy()
        element_onehot = _onehot_encode(elements, self.atom_labels)
        if self.label_idx is not None:
            label = torch.tensor([data['labels'][self.label_idx]])
        else:
            label = torch.tensor([data['label']])

        out_data = Data(x=element_onehot, y=label, pos=pos)
        return out_data


class FullyConnectedEdgeFeaturization:
    def __call__(self, data):
        N = data['pos'].shape[-2]
        pidx_a = torch.outer(torch.arange(N), torch.ones(N))
        pair_indices = torch.stack([pidx_a, torch.outer(torch.ones(N), torch.arange(N))])
        pair_indices = torch.flatten(pair_indices, start_dim=1, end_dim=2)
        pair_vals = torch.ones((N * N, 1))
        pair_indices, pair_vals = _prune_self_loops(pair_indices, pair_vals)
        data['edge_index'] = pair_indices
        data['edge_attr'] = pair_vals
        return data


class DistanceBasisFeaturization:
    """
    Construct rotation-invariant pairwise edge features based on the values of
    distances between points.
    """

    def __init__(self, basis_fxn, **kwargs):
        self.bf_args = kwargs
        self.bf_fxn = basis_fxn

    def __call__(self, data):
        pos = data['pos']
        N = pos.shape[-2]
        displacement_vecs = pos.unsqueeze(-2) - pos.unsqueeze(-3)
        distances = torch.linalg.norm(displacement_vecs, dim=-1)
        features = self.bf_fxn(distances, **self.bf_args)
        # displacement_vecs *= features

        # Flatten into torch_geometric format.
        pidx_a = torch.outer(torch.arange(N), torch.ones(N))
        pair_indices = torch.stack([pidx_a, torch.outer(torch.ones(N), torch.arange(N))])
        pair_indices = torch.flatten(pair_indices, start_dim=1, end_dim=2)
        features = torch.flatten(features, start_dim=-3, end_dim=-2)
        data['edge_index'] = pair_indices
        data['edge_attr'] = features
        return data


def radial_gaussian(distances, r0s, alpha=1.):
    """
    Evaluates Gaussians on the radial values of the distances.
    """
    gauss_arg = distances.unsqueeze(-1) - torch.unsqueeze(r0s, dim=-2)
    basis_fxn = torch.exp(-alpha * gauss_arg**2)
    return basis_fxn


def hard_cutoff(self, distances, r0s, alpha=1.):
    """
    Evaluates Gaussians on the radial values of the distances.
    """
    gauss_arg = distances.unsqueeze(-1) - torch.unsqueeze(r0s, dim=-2)
    basis_fxn = torch.exp(-alpha * gauss_arg**2)
    return basis_fxn


def _onehot_encode(words, labels):
    labels = list(labels)
    return(torch.array([labels.index(i) for i in words]))
