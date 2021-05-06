import numpy as np
import torch
import pygelib_cpp as pcpp
from torch.autograd import Function
from pygelib.utils import _initialize_in_SO3part_view, _convert_to_SO3part_view
from pygelib.SO3VecArray import SO3VecArray, _get_fragment_dict, _get_adims


def cg_product(A, B, output_info=None, lmin=0, lmax=None):
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
    # Ensure everything is in a GElib acceptable format on the backend.
    A_list = [_convert_to_SO3part_view(a) for a in A]
    B_list = [_convert_to_SO3part_view(b) for b in B]

    if output_info is None:
        output_info = _compute_output_shape(A_list, B_list, lmin, lmax)

    num_A = len(A)
    all_tensors = A_list + B_list
    out_tensors = _raw_cg_product.apply(num_A, output_info, lmin, lmax, *all_tensors)
    return SO3VecArray(out_tensors)


class _raw_cg_product(Function):
    """
    Backend cg product routine that interfaces with pytorch's autograd.
    """
    @staticmethod
    # def forward(ctx, A, B, output_info=None, lmin=0, lmax=None):
    def forward(ctx, num_A, output_info, lmin, lmax, *tensors):
        A = tensors[:num_A]
        B = tensors[num_A:]

        # Use pytorch's interface for tensors
        ctx.save_for_backward(*tensors)

        # Save nontensor info normally
        ctx.output_info = output_info
        ctx.num_A = num_A
        ctx.lmin = lmin
        ctx.lmax = lmax

        out = _cg_product_forward(A, B, output_info, lmin, lmax)
        return out

    @staticmethod
    def backward(ctx, *grad_output):
        num_A = ctx.num_A
        A_tensors = ctx.saved_tensors[:num_A]
        B_tensors = ctx.saved_tensors[num_A:]
        A_grad, B_grad = _cg_product_backward(A_tensors, B_tensors, grad_output, ctx.output_info, ctx.lmin, ctx.lmax)
        out_grad = list(A_grad) + list(B_grad)
        return None, None, None, None, *out_grad


def _cg_product_forward(A, B, output_info=None, lmin=0, lmax=None):
    """
    Forward pass for a CG product, A x B

    Parameters
    ----------
    A : list of tensors
        Tensors that make up the first SO3VecArray
    B : list of tensors
        Tensors that make up the second SO3VecArray
    output_info : tuple, None
        Tuple that is output by _compute_output_shape.
    lmin : int, optional
        Minimum output l value to compute. Defaults to 0
    lmax : int, optional
        Maximum output l value to compute.  Default (None) considers
        all possible values.

    Returns
    -------
    product : list of tensors
        list containing the tensors that make up the product the SO3VecArray containing the result of the CG product
    """
    A_ells = [(a.shape[-2] - 1) // 2 for a in A]
    B_ells = [(b.shape[-2] - 1) // 2 for b in B]

    if lmax is None:
        max_l_A = np.max(A_ells)
        max_l_B = np.max(B_ells)
        lmax = max_l_A + max_l_B

    if output_info is None:
        output_keys, output_shapes = _compute_output_shape(A, B, lmin, lmax)
    else:
        output_keys, output_shapes = output_info

    device = A[0].device

    # Create tensors
    def init_fxn(x):
        out = torch.zeros(x, device=device)
        return out

    output_tensors = []
    read_ls = []
    for l, shape in output_shapes.items():
        if l > lmax:
            continue

        output_tensor = _initialize_in_SO3part_view(shape, init_fxn, -2)
        offset = 0  # Start index of next block
        for part_A in A:
            l_A = (part_A.shape[-2] - 1) // 2
            part_A_prt = pcpp._internal_SO3partArray_from_Tensor(part_A[0], part_A[1])
            for part_B in B:
                l_B = (part_B.shape[-2] - 1) // 2
                if (l_A, l_B, l) in output_keys.keys():
                    part_B_prt = pcpp._internal_SO3partArray_from_Tensor(part_B[0], part_B[1])

                    block = output_tensor
                    block_prt = pcpp._internal_SO3partArray_from_Tensor(block[0], block[1])
                    pcpp.add_in_partArrayCGproduct(block_prt, part_A_prt, part_B_prt, offset)
                    offset += output_keys[(l_A, l_B, l)]

        output_tensors.append(output_tensor)
        read_ls.append(l)
    idx = np.argsort(read_ls)
    output_tensors = [output_tensors[i] for i in idx]
    return tuple(output_tensors)


def _cg_product_backward(A, B, product_grad, output_info=None, lmin=0, lmax=None):
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
    device = A[0].device

    A_ells = [(a.shape[-2] - 1) // 2 for a in A]
    B_ells = [(b.shape[-2] - 1) // 2 for b in B]

    if lmax is None:
        max_l_A = np.max(A_ells)
        max_l_B = np.max(B_ells)
        lmax = max_l_A + max_l_B

    if output_info is None:
        output_keys, output_shapes = _compute_output_shape(A, B, lmin, lmax)
    else:
        output_keys, output_shapes = output_info

    def init_fxn(x):
        out = torch.zeros(x, device=device)
        return out

    A_shapes = [a.shape for a in A]
    B_shapes = [b.shape for b in B]

    A_grad_tensors = [_initialize_in_SO3part_view(shape, init_fxn, -2) for shape in A_shapes]
    B_grad_tensors = [_initialize_in_SO3part_view(shape, init_fxn, -2) for shape in B_shapes]

    for k, part_out_grad in enumerate(product_grad):
        l_out = (part_out_grad.shape[-2] - 1) // 2

        block_start = 0  # Start index of next block
        for i, part_A in enumerate(A):
            l_A = (part_A.shape[-2] - 1) // 2

            # Cast A, A_grad to gelib backend
            A_i_gelib = pcpp._internal_SO3partArray_from_Tensor(part_A[0], part_A[1])
            Agt_i = A_grad_tensors[i]
            A_grad_i_gelib = pcpp._internal_SO3partArray_from_Tensor(Agt_i[0], Agt_i[1])

            for j, part_B in enumerate(B):
                l_B = (part_B.shape[-2] - 1) // 2

                B_j_gelib = pcpp._internal_SO3partArray_from_Tensor(part_B[0], part_B[1])
                Bgt_j = B_grad_tensors[j]
                B_grad_j_gelib = pcpp._internal_SO3partArray_from_Tensor(Bgt_j[0], Bgt_j[1])

                if (l_A, l_B, l_out) in output_keys.keys():
                    block_end = block_start + output_keys[(l_A, l_B, l_out)]
                    grad_block = _convert_to_SO3part_view(part_out_grad[..., block_start:block_end])

                    # Convert everything to GElib tensors
                    grad_block_gelib = pcpp._internal_SO3partArray_from_Tensor(grad_block[0], grad_block[1])

                    pcpp.add_in_partArrayCGproduct_back0(A_grad_i_gelib, grad_block_gelib, B_j_gelib, 0)
                    pcpp.add_in_partArrayCGproduct_back1(B_grad_j_gelib, grad_block_gelib, A_i_gelib, 0)

                    block_start = block_end
        # return SO3VecArray(A_grad_tensors), SO3VecArray(B_grad_tensors)
    return tuple(A_grad_tensors), tuple(B_grad_tensors)


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
    rdim = -2
    A_fragment_dict = _get_fragment_dict(A, rdim)
    B_fragment_dict = _get_fragment_dict(B, rdim)

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
            output_shape_l.insert(-1, 2 * l + 1)
            # else:
            #     output_shape_l.insert(B.rdim, 2 * l + 1)
            total_output_shapes[l] = output_shape_l

    return output_channels, total_output_shapes


def _check_product_is_possible(A, B):
    A_arr_shapes = _get_adims(A)
    B_arr_shapes = _get_adims(B)

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
