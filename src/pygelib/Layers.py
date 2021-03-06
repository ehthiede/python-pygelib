from pygelib.SO3VecArray import _get_tensor_ell, SO3VecArray
from pygelib.spharm import pos_to_rep
from pygelib.CG_routines import cg_product
from pygelib import utils
from torch_scatter import scatter
import numpy as np
import torch


class ManyEdgeMPLayer(torch.nn.Module):
    def __init__(self, real_edges=True):
        super(ManyEdgeMPLayer, self).__init__()
        self.real_edges = real_edges

    def forward(self, X, edge_vals, edge_idx):
        """
        Runs forward pass of the Message passing layer.

        Args:
            x :
        """
        out_tensors = []
        for x in X:
            N = x.shape[1]
            x_out_tensors = []
            for e_j in edge_vals.unbind(-1):  # Iterate over edge channels.
                x_out_j = utils._complex_spmm(edge_idx, e_j, N, x,
                                              self.real_edges)
                x_out_tensors.append(x_out_j)
            out_tensors.append(torch.cat(x_out_tensors, dim=-1))

        return SO3VecArray(out_tensors, rdim=X.rdim)


class L1DifferenceLayer(torch.nn.Module):
    """
    Builds some initial l=0 features
    """

    def __init__(self, basis_fxn, remove_self_loops=True, **kwargs):
        super(L1DifferenceLayer, self).__init__()
        self.bf_args = kwargs
        self.bf_fxn = basis_fxn
        self.remove_self_loops = remove_self_loops
        self.expand_node_feats = ExpandNodeLayer()

    def forward(self, pos, node_features, edge_idx):
        """
        Constructs initial l=1 features from invariants and positions
        """
        # Evaluate basis functions.
        N = len(pos)
        if self.remove_self_loops:
            edge_idx, __ = utils._prune_self_loops(edge_idx)
        displacement = pos[edge_idx[1]] - pos[edge_idx[0]]  # M (num edges) x 3

        distances = torch.linalg.norm(displacement, dim=-1)  # M
        bf_values = self.bf_fxn(distances, **self.bf_args)  # M x C
        bf_vecs = displacement / distances.unsqueeze(-1)
        bf_vecs = bf_vecs.unsqueeze(-1) * bf_values.unsqueeze(-2)  # M x 3 x C

        # Multiply by Node Features
        
        nf_edge = node_features[edge_idx[1]]  # M x D
        bf_vecs = bf_vecs.unsqueeze(-1) * nf_edge.unsqueeze(-2).unsqueeze(-2)
        bf_vecs = torch.flatten(bf_vecs, start_dim=-2)  # M x 3 x (D*C)
        
        # Scatter back to individual nodes and convert to spherical tensor.
        node_vecs = scatter(bf_vecs, edge_idx[0], dim=0, dim_size=N)
        node_vecs = pos_to_rep(node_vecs, xyzdim=-2)

        # Build some node features
        scalar_node_feats = self.expand_node_feats(node_features)
        scalar_node_feats = scalar_node_feats[0]

        return SO3VecArray([scalar_node_feats, node_vecs])


class ExpandNodeLayer(torch.nn.Module):
    """
    Expands Node features into an l=0 feature
    """

    def __init__(self, is_complex: bool = False):
        super(ExpandNodeLayer, self).__init__()
        self.is_complex = is_complex

    def forward(self, node_features):
        if not self.is_complex:
            nf_imag = torch.zeros_like(node_features)
            node_features = torch.stack([node_features, nf_imag], dim=0)
        node_features = node_features.unsqueeze(-2)
        return SO3VecArray([node_features])


class SO3Linear(torch.nn.Module):
    """
    Layer that performs a linear mix

    Parameters
    ----------
    transformation_dict : dict
        Dictionary giving input and output channels for each l.
        Keys are integers corresponding to value of l, values are
        a tuple of (channels_in, channels_out).
    """

    def __init__(self, transformation_dict):
        super(SO3Linear, self).__init__()

        self.lin_layers_real = torch.nn.ModuleDict()
        self.lin_layers_imag = torch.nn.ModuleDict()
        for l, (c_in, c_out) in transformation_dict.items():
            self.lin_layers_real[str(int(l))] = torch.nn.Linear(c_in, c_out, bias=False)
            self.lin_layers_imag[str(int(l))] = torch.nn.Linear(c_in, c_out, bias=False)

    def forward(self, X):
        out_tensors = []
        ells = []
        for x in X:
            ell_x = _get_tensor_ell(x, X.rdim)
            lin_real = self.lin_layers_real[str(ell_x)]
            lin_imag = self.lin_layers_imag[str(ell_x)]

            x_out_real = lin_real(x[0]) - lin_imag(x[1])
            x_out_imag = lin_real(x[1]) + lin_imag(x[0])
            # x_out_real = lin_real(x[0])
            # x_out_imag = lin_real(x[1])

            out_tensors.append(torch.stack([x_out_real, x_out_imag], dim=0))
            ells.append(ell_x)

        # Sort for convenience
        ls = np.array(ells)
        idx = np.argsort(ls)
        out_tensors = [out_tensors[i] for i in idx]
        X_out = SO3VecArray(out_tensors, X.rdim)

        return X_out


class CGProduct(torch.nn.Module):
    """
    Layer that performs the CG product nonlinearity on two SO3VecArray
    """

    def __init__(self, output_info=None, lmin=0, lmax=None):
        super(CGProduct, self).__init__()
        self.output_info = output_info
        self.lmin = lmin
        self.lmax = lmax

    def forward(self, A, B):
        return cg_product(A, B, self.output_info, self.lmin, self.lmax)
