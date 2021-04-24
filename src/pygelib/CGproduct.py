import numpy as np
import torch
import pygelib_cpp as pcpp
from pygelib.utils import _initialize_in_SO3part_view
from pygelib.SO3VecArray import SO3VecArray


def cg_product_forward(A, B, output_info=None, lmin=0, lmax=None):
    """
    Forward pass for a CG product, A x B

    Parameters
    ----------
    A : SO3VecArray
        First SO3VecArray
    B : SO3VecArray
        Second SO3VecArray
    output_info : tuple, None
        Tuple that is output by _compute_output_shape.
    lmin : int, optional
        Minimum output l value to compute. Defaults to 0
    lmax : int, optional
        Maximum output l value to compute.  Default (None) considers
        all possible values.

    Returns
    -------
    product : SO3VecArray
        SO3VecArray containing the result of the CG product
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

    print(output_keys)
    print(output_shapes)

    output_tensors = []
    read_ls = []
    for l, shape in output_shapes.items():
        if l > lmax:
            continue

        output_tensor = _initialize_in_SO3part_view(shape, init_fxn, A.rdim)
        offset = 0  # Start index of next block
        for part_A in A:
            l_A = (part_A.shape[A.rdim] - 1) // 2
            part_A_prt = pcpp._internal_SO3partArray_from_Tensor(part_A[0], part_A[1])
            for part_B in B:
                l_B = (part_B.shape[B.rdim] - 1) // 2
                if (l_A, l_B, l) in output_keys.keys():
                    part_B_prt = pcpp._internal_SO3partArray_from_Tensor(part_B[0], part_B[1])

                    # block_size = block_start + output_keys[(l_A, l_B, l)]

                    # block = output_tensor[..., block_start:block_end]
                    block = output_tensor
                    print("-------------------------")
                    print("Editing l=%d, offset=%d" % (l, offset))
                    print(block)
                    block_prt = pcpp._internal_SO3partArray_from_Tensor(block[0], block[1])
                    # if offset > 0:
                    #     if l > 0:
                    #         print(offset, l)
                    #         print(block.shape)
                    #         raise Exception
                    pcpp.add_in_partArrayCGproduct(block_prt, part_A_prt, part_B_prt, offset)

                    # Update offset
                    offset += output_keys[(l_A, l_B, l)]
                    # offset += 1
                    # block_start = block_end
                    print("Finished editing block: edited block:")
                    print(block)

        output_tensors.append(output_tensor)
        read_ls.append(l)
        print("Done with l=", l)
    idx = np.argsort(read_ls)
    output_tensors = [output_tensors[i] for i in idx]
    return SO3VecArray(output_tensors)


def cg_product_backward(A, B, product_grad, output_info=None, lmin=0, lmax=None):
    """
    Evaluates the backward derivative of the CG product A x B.

    Parameters
    ----------
    A : SO3VecArray
        First SO3VecArray used to construct the product.
    B : SO3VecArray
        Second SO3VecArray used to construct the product.
    product_grad : SO3VecArray
        SO3VecArray containing the gradient associated with entry in the A x B.
    output_info : tuple, None
        Tuple that is output by _compute_output_shape.
    lmin : int, optional
        Minimum output l value to compute. Defaults to 0
    lmax : int, optional
        Maximum output l value to compute.  Default (None) considers
        all possible values.

    Returns
    -------
    A_grad : SO3VecArray
        SO3VecArray containing the gradient against each element of A.
    B_grad : SO3VecArray
        SO3VecArray containing the gradient against each element of B.
    """
    if lmax is None:
        max_l_A = np.max(A.ells)
        max_l_B = np.max(B.ells)
        lmax = max_l_A + max_l_B

    def init_fxn(x):
        out = torch.zeros(x, device=device)
        return out

    A_grad_tensors = [_initialize_in_SO3part_view(shape, init_fxn, A.rdim) for shape in A.shapes]
    B_grad_tensors = [_initialize_in_SO3part_view(shape, init_fxn, B.rdim) for shape in B.shapes]

    for grad_part in product_grad:
        grad_part_l = (grad_part[product_grad.rdim] - 1) // 2

        block_start = 0  # Start index of next block
        # for A_prt, A_grad_pt in zip(A, A_grad_tensors):
        #     for B_prt, B_grad_pt in zip(B, B_grad_tensors):
        for i, A_prt in enumerate(A):
            l_A = (part_A.shape[A.rdim] - 1) // 2
            for j, B_prt in enumerate(B):
                l_B = (part_B.shape[B.rdim] - 1) // 2
                if (l_A, l_B, l) in output_keys.keys():
                    block_end = block_start + output_keys[(l_A, l_B, l)]
                    grad_block = output_tensor[..., block_start:block_end]
                    pcpp.add_in_partArrayCGproduct_back0(A_grad_tensors[i], grad_block, B)
                    pcpp.add_in_partArrayCGproduct_back0(B_grad_tensors[j], grad_block, A)
        return SO3VecArray(A_grad_tensors), SO3VecArray(B_grad_tensors)


def _compute_output_shape(A, B, lmin=0, lmax=None):
    """
    Computes the shapes for the output of a CGproduct between SO3VecArrays A and B.

    Parameters
    ----------
    A : SO3VecArray
        First SO3VecArray in the product.
    B : SO3VecArray
        Second SO3VecArray in the product.
    lmin : int, optional
        Minimum ell value to compute the product for
    lmax : int, optional
        Maximum ell value to compute the product for

    Returns
    -------
    output_channels: dict
        Dict with the number of channels associated with each product.
        Each key is a tuple (l_A, l_B, l_out) and the associated value
        is the number of channels associated with the product.
        Note this does *not* account for l_max.
    total_output_shapes: dict
        Dict with the number of channels associated with each output l.
        Note this *is* truncated by l_max.
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

    output_channels = {}
    for l_A, nc_A in A_fragment_dict.items():
        type_A = np.zeros(l_A + 1, dtype=int)
        type_A[-1] = nc_A
        for l_B, nc_B in B_fragment_dict.items():
            type_B = np.zeros(l_B + 1, dtype=int)
            type_B[-1] = nc_B
            type_AB = pcpp.estimate_num_products(list(type_A), list(type_B))
            for l_out, nc_out in enumerate(type_AB):
                if nc_out != 0:
                    output_channels[(l_A, l_B, l_out)] = nc_out

    total_output_shapes = {}
    for (__, ___, l), nc in output_channels.items():
        if ((l > lmax) or (l < lmin)):
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

    return output_channels, total_output_shapes


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
