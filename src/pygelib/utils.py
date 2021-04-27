import numpy as np
import torch
from math import ceil
from copy import deepcopy


def move_from_end(x, dim):
    """
    Moves the last dimension to the desired position while preserving ordering of the others.

    Parameters
    ----------
    x : :class:`torch.Tensor`
        Input tensor of shape ... x n
    dim : int
        Dimension to move the last location.

    Returns
    -------
    x : :class:`torch.Tensor`
        Permuted tensor
    """
    N = len(x.shape)
    permute_indices = list(range(N-1))
    permute_indices.insert(dim, N-1)
    return x.permute(permute_indices)


def move_to_end(x, dim):
    """
    Moves a specified dimension to the end.

    """
    N = len(x.shape)
    permute_indices = list(range(N))
    permute_indices.remove(dim)
    permute_indices.append(dim)
    return x.permute(permute_indices)


def _initialize_in_SO3part_view(shape, init_fxn, cell_index=-2, padding_multiple=32):
    adims = tuple(shape[:cell_index])  # Indices of cells
    cdims = tuple(shape[cell_index:])  # Indices inside a cell
    flattened_cdims = np.prod(cdims)
    padded_cdim_size = padding_multiple * ceil(flattened_cdims / padding_multiple)

    init_dims = adims + (padded_cdim_size,)
    output_data = init_fxn(init_dims)
    new_strides = _get_expanded_strides(adims, cdims, padded_cdim_size)
    output = torch.as_strided(output_data, shape, new_strides)
    return output


def _get_expanded_strides(adims, cdims, padded_cdim_size):
    reverse_strides = []
    init_dim = 1
    for d in cdims[::-1]:
        reverse_strides.append(init_dim)
        init_dim *= d
    init_dim = padded_cdim_size
    for d in adims[::-1]:
        reverse_strides.append(init_dim)
        init_dim *= d
    return tuple(reverse_strides[::-1])


def _convert_to_SO3part_view(tensor, cell_index=-2, padding_multiple=32):
    """
    Converts a tensor corresponding to an SO3part into a view that is acceptable
    by GElib.
    """
    beginning_strides = tensor.stride()[:cell_index]

    is_in_acceptable_view = True
    for i, stride in enumerate(beginning_strides):
        if (stride % padding_multiple != 0):
            is_in_acceptable_view = False
            break

    if is_in_acceptable_view:
        return tensor
    else:
        # print('making contiguous')
        # tensor = tensor.contiguous()
        # print('calling _copy_into_SO3part_view')
        return _copy_into_SO3part_view(tensor.contiguous(), cell_index, padding_multiple)


def _copy_into_SO3part_view(tensor, cell_index=-2, padding_multiple=32):
    shape = tensor.shape
    if cell_index < 0:
        cell_index = len(shape) + cell_index
    adims = shape[:cell_index]  # Indices of cells
    cdims = shape[cell_index:]  # Indices inside a cell

    # Flatten indices in the cell.
    flattened_cdims = np.prod(cdims)
    old_flat_shape = adims + (flattened_cdims,)
    # print(tensor.shape, old_flat_shape)
    # print(tensor.stride())
    # print(tensor.contiguous().stride())
    flat_tensor = tensor.view(old_flat_shape)

    # Pad tensor so cell shapes are multiple of padding_multiple
    padded_cdim_size = padding_multiple * ceil(flattened_cdims / padding_multiple)
    diff = padded_cdim_size - flattened_cdims
    padded_tensor = torch.nn.functional.pad(flat_tensor, (0, diff), 'constant', 0)

    # Calculate strides for the padded tensor
    old_strides = tensor.stride()
    new_strides = deepcopy(list(old_strides))
    for i in range(cell_index):
        assert((old_strides[i] % flattened_cdims) == 0)
        # old_factor = old_strides[i]  // flattened_cdims
        new_strides[i] = (old_strides[i] // flattened_cdims) * padded_cdim_size
    new_strides = tuple(new_strides)

    # reconstruct the original tensor as a view of the expanded tensor.
    output_tensor = torch.as_strided(padded_tensor, shape, new_strides)
    return output_tensor
