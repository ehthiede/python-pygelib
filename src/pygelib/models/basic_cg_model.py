import torch
import numpy as np
from torch_scatter import scatter
from pygelib_cpp import estimate_num_products
from pygelib.Layers import SO3Linear, ManyEdgeMPLayer, L1DifferenceLayer, CGProduct
from pygelib.transforms import radial_gaussian


class CGMPModel(torch.nn.Module):
    def __init__(self, nc_in: int, n_channels: int, n_layers: int, lmax: int, r0s: torch.Tensor):
        super(CGMPModel, self).__init__()
        num_edge_channels = len(r0s)
        self.initial_layer = L1DifferenceLayer(radial_gaussian, r0s=r0s, alpha=0.0)
        self.initial_lin_layer = SO3Linear({1: (nc_in * num_edge_channels, n_channels)})
        self.final_lin_layer = SO3Linear({0: (n_channels, 1)})

        self.cg_mp_blocks = torch.nn.ModuleList()
        lm = min(lmax, 2)
        for i in range(n_layers):
            transformation_dict = {k: (n_channels, n_channels) for k in range(lm)}
            block = CGMPBlock(transformation_dict, num_edge_channels, real_edges=True)
            self.cg_mp_blocks.append(block)
            lm = min(lmax, lm**2)

    def forward(self, data):
        x = data['x']
        edge_vals = data['edge_attr']
        edge_idx = data['edge_attr']
        x = self.initial_layer(data['pos'], x, edge_idx)
        x = self.initial_lin_layer(x)
        for layer in self.cg_mp_blocks:
            x = layer(x, edge_vals, edge_idx)
        x = self.final_lin_layer(x)[0]
        x = x.unsqueeze(-2).unsqueeze(-1) # Remove unnecessary channels
        x = scatter(x, data['batch'], dim=0, reduce='mean')
        return x.unsqueeze(-2).unsqueeze(-1)


class CGMPBlock(torch.nn.Module):
    def __init__(self, transformation_dict, num_edge_channels, real_edges=True):
        """
        Basic message passing block followed by a nonlinearity.

        Args:
            transformation_dict (dict): Dictionary giving input and output
            channels for each l.  Keys are integers corresponding to value of l,
            values are a tuple of (channels_in, channels_out).
        """
        super(CGMPBlock, self).__init__()

        # construct linear transformation_sizes
        lmin = min(list(transformation_dict.keys()))
        lmax = max(list(transformation_dict.keys()))
        input_ls = np.zeros(lmax + 1, dtype=int)
        # Determine size of product
        for l, (nc_in, __) in transformation_dict.items():
            input_ls[l] = nc_in
        xx_product_ells = estimate_num_products(input_ls, input_ls)
        yx_product_ells = estimate_num_products(num_edge_channels * input_ls, input_ls)
        yy_product_ells = estimate_num_products(num_edge_channels * input_ls,
                                                num_edge_channels * input_ls)

        # Construct input and output sizes required for the linear layer.
        xx_linear_dict = {}
        yx_linear_dict = {}
        yy_linear_dict = {}
        for l, (__, nc_out) in transformation_dict.items():
            nc_prod = xx_product_ells[l]
            if nc_prod == 0:
                raise Exception("Trying to construct output feature with no ell's from CG product!")
            xx_linear_dict[l] = (nc_prod, nc_out)

            nc_prod = yx_product_ells[l]
            yx_linear_dict[l] = (nc_prod, nc_out)
            nc_prod = yy_product_ells[l]
            yy_linear_dict[l] = (nc_prod, nc_out)

        self.mp_layer = ManyEdgeMPLayer(real_edges)
        # self.cg_layer = CGProduct(lmin=lmin, lmax=lmax+10)
        self.cg_layer = CGProduct(lmin=lmin, lmax=lmax)
        self.lin_layer_id_id = SO3Linear(xx_linear_dict)
        self.lin_layer_mp_id = SO3Linear(yx_linear_dict)
        self.lin_layer_mp_mp = SO3Linear(yy_linear_dict)

    def forward(self, x, edge_vals, edge_idx):
        y = self.mp_layer(x, edge_vals, edge_idx)
        yy = self.cg_layer(y, y)
        y_mp_mp = self.lin_layer_mp_mp(yy)
        y_mp_id = self.lin_layer_mp_id(self.cg_layer(y, x))
        y_id_id = self.lin_layer_id_id(self.cg_layer(x, x))
        return y_mp_mp + y_mp_id + y_id_id
