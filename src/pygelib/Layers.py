from pygelib.SO3VecArray import _get_tensor_ell, SO3VecArray
from pygelib.CG_routines import cg_product
import numpy as np
import torch


class Linear(torch.nn.Module):
    """
    Layer that performs a  linera mix
    """
    def __init__(self, transformations):
        super(Linear, self).__init__()

        self.lin_layers_real = torch.nn.ModuleDict()
        self.lin_layers_imag = torch.nn.ModuleDict()
        for l, (c_in, c_out) in transformations.items():
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
