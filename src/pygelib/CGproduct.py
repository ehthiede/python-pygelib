import numpy as np
import pygelib_cpp as backend
from utils import


def cg_product_forward(A, B, output_shapes=None, lmin=0, lmax=None):
    """
    Forward pass for a CG product, A x B

    A : SO3vecArray
        First SO3vecArray
    B : SO3vecArray
        Second SO3vecArray

    """
    # Create tensors
    if output_shapes is None:
        output_shapes = _compute_output_shape(A, B, lmin, lmax)


    # Create output
    # output_tensors = {l: torch


def _compute_output_shape(A, B, lmin=0, lmax=None):
    """
    Computes the shapes for the output of a CGproduct between SO3VecArrays A and B.
    """
    A_fragment_dict = A.fragment_dict
    B_fragment_dict = B.fragment_dict
    output_adim = _check_product_is_possible(A, B)
    A_ells = A_fragment_dict.keys()
    B_ells = B_fragment_dict.keys()

    if lmax is None:
        max_l_A = np.max(A_ells)
        max_l_B = np.max(B_ells)
        lmax = max_l_A + max_l_B

    output_shapes = {}
    for l_A, nc_A in A_fragment_dict.items():
        for l_B, nc_B in B_fragment_dict.items():
            possible_lmin = max(abs(l_A - l_B), lmin)
            possible_lmax = min(abs(l_A + l_B), lmin)
            for l_out in range(possible_lmin, possible_lmax):
                num_channels = nc_A * nc_B
                if l_out in output_shapes.keys():
                    output_shapes[l_out][-1] += num_channels
                else:
                    output_shape_l = list(output_adim) + [num_channels]
                    output_shape_l.insert(B.rdim, 2 * l_out + 1)
                    output_shapes[l_out] = output_shape_l

    return output_shapes


def _check_product_is_possible(A, B):
    A_arr_shapes = [aa for aa in A.adims]
    B_arr_shapes = [bb for bb in B.adims]

    for adim_i in A_arr_shapes[1:]:
        assert(adim_i == A_arr_shapes[0]), "Tensors in SO3PartArray A have different array dimensions."

    for adim_i in B_arr_shapes[1:]:
        assert(adim_i == B_arr_shapes[0]), "Tensors in SO3PartArray B have different array dimensions."

    _check_multiplyable(A_arr_shapes[0], B_arr_shapes[0])
    return list(A_arr_shapes[0])


def _check_multiplyable(A_arr_shape, B_arr_shape):
    assert(A_arr_shape == B_arr_shape), "Tensors in A and B have different array dimensions\
                                         Currently no other options are multiplyable..."

# if l_A + l_B > lmax:
#     continue
# if abs(l_A - l_B) < lmin:
#     continue

# def old_stuff():
#     A_fragment_dict = A.fragment_dict
#     B_fragment_dict = B.fragment_dict
