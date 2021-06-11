import torch
from pygelib_cpp import estimate_num_products
import numpy as np
from pygelib.Layers import SO3Linear, ManyEdgeMPLayer, L1DifferenceLayer, CGProduct


class CGMPBlock(torch.nn.Module):
    def __init__(self, transformation_dict, real_edges=True):
        """
        Basic message passing block followed by a nonlinearity.

        Args: 
            transformation_dict (dict): Dictionary giving input and output 
            channels for each l.  Keys are integers corresponding to value of l,
            values are a tuple of (channels_in, channels_out).
        """
        super(CGMPBlock, self).__init__()

        # construct linear transformation_sizes
        lmin = min([transformation_dict.keys()])
        lmax = max([transformation_dict.keys()])
        input_ls = np.zeros(lmax + 1, dtype=int)
        # Determine size of product
        for l, (nc_in, __) in transformation_dict.items():
            input_ls[l] = nc_in
        product_ells = estimate_num_products(input_ls, input_ls)

        # Construct input and output sizes required for the linear layer.
        linear_transformation_dict = {}
        for l, (__, nc_out) in transformation_dict.items():
            nc_prod = product_ells[l]
            if nc_prod == 0:
                raise Exception("Trying to construct output feature with no ell's from CG product!")
            linear_transformation_dict[l] = (nc_prod, nc_out)

        self.mp_layer = ManyEdgeMPLayer(real_edges)
        self.cg_layer = CGProduct(lmin=lmin, lmax=lmax)
        self.lin_layer_mp_mp = SO3Linear(linear_transformation_dict)
        self.lin_layer_mp_id = SO3Linear(linear_transformation_dict)
        self.lin_layer_id_id = SO3Linear(linear_transformation_dict)

    def forward(self, x, edge_vals, edge_idx):
        y = ManyEdgeMPLayer(x, edge_vals, edge_idx)
        y_mp_mp = self.lin_layer_mp_mp(self.cg_layer(y, y))
        y_mp_id = self.lin_layer_mp_id(self.cg_layer(y, x))
        y_id_id = self.lin_layer_id_id(self.cg_layer(x, x))
        return y_mp_mp + y_mp_id + y_id_id + x
