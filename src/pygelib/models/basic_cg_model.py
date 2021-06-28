import torch
import numpy as np
from torch_scatter import scatter
from pygelib_cpp import estimate_num_products
from pygelib.Layers import SO3Linear, ManyEdgeMPLayer, L1DifferenceLayer, CGProduct
from pygelib.transforms import radial_gaussian


class CGMPModel(torch.nn.Module):
    def __init__(self, nc_in_node: int, nc_edge: int, n_channels: int, n_layers: int, lmax: int, r0s: torch.Tensor):
        super(CGMPModel, self).__init__()
        self.initial_layer = L1DifferenceLayer(radial_gaussian, r0s=r0s, alpha=0.0)
        self.initial_lin_layer = SO3Linear({0: (nc_in_node, n_channels), 1: (nc_in_node * len(r0s), n_channels)})

        self.cg_mp_blocks = torch.nn.ModuleList()

        lm = min(lmax, 1)
        for i in range(n_layers-2):
            transformation_dict = {k: (n_channels, n_channels) for k in range(lm+1)}
            lm_new = min(lmax, 2 * lm)
            new_channels = {k: (0, n_channels) for k in range(lm+1, lm_new+1)}
            transformation_dict.update(new_channels)
            block = CGMPBlock(transformation_dict, nc_edge, real_edges=True)
            self.cg_mp_blocks.append(block)
            lm = lm_new
        transformation_dict = {0: (n_channels, n_channels)}
        transformation_dict.update({i: (n_channels, 0) for i in range(1, lm+1)})
        final_block = CGMPBlock(transformation_dict, nc_edge, real_edges=True)
        self.cg_mp_blocks.append(final_block)

        self.final_lin = torch.nn.Linear(2 * n_channels, 1, bias=False)

    def forward(self, data):
        x = data['x']
        edge_vals = data['edge_attr']
        edge_idx = data['edge_idx']
        x = self.initial_layer(data['pos'], x, edge_idx)
        x = self.initial_lin_layer(x)
        for layer in self.cg_mp_blocks:
            x = layer(x, edge_vals, edge_idx)
        # x = self.final_lin_layer(x)[0]
        x = x[0].squeeze(-2).squeeze(-1)  # Remove unnecessary channels
        x = torch.cat([x[0], x[1]], dim=-1) # Move real, imaginary to channel indices
        x = self.final_lin(x).squeeze(-1)

        x = scatter(x, data['batch'], dim=0, reduce='mean')

        # Concatenate real and imaginary and mix.
        return x


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
        output_ls = [k for k, v in transformation_dict.items() if v[1] > 0]
        lmax_in = max(list(transformation_dict.keys()))
        lmin = min(output_ls)
        lmax = max(output_ls)
        input_ls = np.zeros(lmax_in + 1, dtype=int)
        # Determine size of product
        for l, (nc_in, __) in transformation_dict.items():
        # for l in output_ls:
            input_ls[l] = nc_in
        xx_product_ells = estimate_num_products(input_ls, input_ls)
        yx_product_ells = estimate_num_products(num_edge_channels * input_ls, input_ls)
        yy_product_ells = estimate_num_products(num_edge_channels * input_ls,
                                                num_edge_channels * input_ls)
        self.transformation_dict = transformation_dict

        # Construct input and output sizes required for the linear layer.
        xx_linear_dict = {}
        yx_linear_dict = {}
        yy_linear_dict = {}
        for l in output_ls:
            nc_out = transformation_dict[l][1]
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
