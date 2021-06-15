from pygelib.SO3VecArray import SO3VecArray
import cormorant
from pygelib.profiling.cg_ops import _CGProduct
from cormorant.cg_lib import CGModule


class CormorantCGProduct(CGModule):
    """
    Layer that performs the CG product nonlinearity on two SO3VecArray
    """
    def __init__(self, tau_A, tau_B, lmin=0, lmax=None, cg_dict=None, enforce_contiguous=True, device=None, dtype=None):
        super().__init__(cg_dict=cg_dict, maxl=lmax, device=device, dtype=None)

        self.lmin = lmin
        self.lmax = lmax
        self.enforce_contiguous = enforce_contiguous

        if cormorant is not None:
            self.CGProductModule = _CGProduct(tau_A, tau_B, maxl=lmax, device=device, cg_dict=cg_dict)
            # self.CGProductModule._init_cg_dict(cg_dict, lmax)
        else:
            raise ImportError("Was unable to find the cormorant package")

    def forward(self, A, B):
        # Transpose to cormorant format
        A_t = [a.permute(1, 3, 2, 0) for a in A]
        B_t = [b.permute(1, 3, 2, 0) for b in B]
        if self.enforce_contiguous:
            A_t = [a.contiguous() for a in A_t]
            B_t = [b.contiguous() for b in B_t]

        corm_A = cormorant.so3_lib.SO3Vec(A_t)
        corm_B = cormorant.so3_lib.SO3Vec(B_t)
        corm_prod = self.CGProductModule(corm_A, corm_B)

        # Transpose back from to pygelib format
        prod = [p.permute(3, 0, 2, 1) for p in corm_prod]
        if self.enforce_contiguous:
            prod = [p.contiguous() for p in prod]

        return SO3VecArray(prod)
