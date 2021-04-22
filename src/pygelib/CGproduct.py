import numpy as np
import torch
import pygelib_cpp as pcpp
from pygelib.utils import _initialize_in_SO3part_view
from pygelib.SO3VecArray import SO3VecArray


def cg_product_forward(A, B, output_info=None, lmin=0, lmax=None):
    """
    Forward pass for a CG product, A x B

    A : SO3VecArray
        First SO3VecArray
    B : SO3VecArray
        Second SO3VecArray

    """
    if lmax is None:
        max_l_A = np.max(A.ells)
        max_l_B = np.max(B.ells)
        lmax = max_l_A + max_l_B

    # Create tensors
    if output_info is None:
        output_keys, output_shapes = _compute_output_shape(A, B, lmin, lmax)
    else:
        output_keys, output_shapes = output_info

    device = A[0].device

    def init_fxn(x):
        out = torch.zeros(x, device=device)
        return out

    output_tensors = []
    read_ls = []
    for l, shape in output_shapes.items():
        if l > lmax:
            continue

        output_tensor = _initialize_in_SO3part_view(shape, init_fxn, A.rdim)
        block_start = 0  # Start index of next block
        for part_A in A:
            l_A = (part_A.shape[A.rdim] - 1) // 2
            part_A_prt = pcpp._internal_SO3partArray_from_Tensor(part_A[0], part_A[1])
            for part_B in B:
                l_B = (part_B.shape[B.rdim] - 1) // 2
                if (l_A, l_B, l) in output_keys.keys():
                    part_B_prt = pcpp._internal_SO3partArray_from_Tensor(part_B[0], part_B[1])

                    block_end = block_start + output_keys[(l_A, l_B, l)]
                    block = output_tensor[..., block_start:block_end]
                    # block = output_tensor
                    block_prt = pcpp._internal_SO3partArray_from_Tensor(block[0], block[1])
                    pcpp.add_in_partArrayCGproduct(block_prt, part_A_prt, part_B_prt)
                    block_start = block_end

        output_tensors.append(output_tensor)
        read_ls.append(l)

    idx = np.argsort(read_ls)
    output_tensors = [output_tensors[i] for i in idx]
    return SO3VecArray(output_tensors)


def _compute_output_shape(A, B, lmin=0, lmax=None):
    """
    Computes the shapes for the output of a CGproduct between SO3VecArrays A and B.
    """
    A_fragment_dict = A.fragment_dict
    B_fragment_dict = B.fragment_dict
    output_adim = _check_product_is_possible(A, B)
    A_ells = list(A_fragment_dict.keys())
    B_ells = list(B_fragment_dict.keys())

    if lmax is None:
        max_l_A = np.max(A_ells)
        max_l_B = np.max(B_ells)
        lmax = max_l_A + max_l_B

    output_keys = {}
    for l_A, nc_A in A_fragment_dict.items():
        type_A = np.zeros(l_A + 1, dtype=int)
        type_A[-1] = nc_A
        for l_B, nc_B in B_fragment_dict.items():
            type_B = np.zeros(l_B + 1, dtype=int)
            type_B[-1] = nc_B
            type_AB = pcpp.estimate_num_products(list(type_A), list(type_B))
            for l_out, nc_out in enumerate(type_AB):
                if nc_out != 0:
                    output_keys[(l_A, l_B, l_out)] = nc_out

    total_output_shapes = {}
    for (__, ___, l), nc in output_keys.items():
        if l > lmax:
            break
        if l in total_output_shapes.keys():
            total_output_shapes[l][-1] += nc
        else:
            output_shape_l = [2] + list(output_adim) + [nc]
            if B.rdim < 0:
                output_shape_l.insert(B.rdim+1, 2 * l + 1)
            else:
                output_shape_l.insert(B.rdim, 2 * l + 1)
            total_output_shapes[l] = output_shape_l

    return output_keys, total_output_shapes


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
