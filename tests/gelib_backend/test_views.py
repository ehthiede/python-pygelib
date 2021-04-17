import pytest
import torch
import numpy as np
from math import ceil
from pygelib import utils


class TestInitialization():
    @pytest.mark.parametrize('torchfxn', [torch.ones, torch.zeros])
    @pytest.mark.parametrize('shapes', [((6, 5, 2, 32), 2),
                                        ((6, 5, 2, 30), 2),
                                        ((5, 3, 33), 1),
                                        ])
    def test_expanded_initialization(self, torchfxn, shapes):
        shape, cell_index = shapes
        output = utils._initialize_in_SO3part_view(shape, torchfxn, cell_index)
        ref_output = torchfxn(shape)

        assert(torch.allclose(ref_output, output))

    @pytest.mark.parametrize('inouts', [((6, 5, 2, 32), (320, 64, 32, 1), 2),
                                        ((6, 5, 2, 30), (320, 64, 30, 1), 2),
                                        ((5, 3, 33), (128, 33, 1), 1),
                                        ])
    def test_get_expanded_strides(self, inouts):
        in_shape, out_strides, cell_index = inouts
        adims = in_shape[:cell_index]
        cdims = in_shape[cell_index:]

        flattened_cdims = np.prod(cdims)
        padded_cdim_size = 32 * ceil(flattened_cdims / 32)

        predicted_strides = utils._get_expanded_strides(adims, cdims, padded_cdim_size)
        assert(out_strides == predicted_strides)


class TestConversion():
    @pytest.mark.parametrize('shapes', [((3, 5, 31), 1),
                                        ((4, 32), 1),
                                        ((2, 2, 33,), 2)])
    @pytest.mark.parametrize('device', [torch.device('cuda'), torch.device('cpu')])
    def test_tensors_match(self, shapes, device):
        shape, cell_index = shapes
        tensor = torch.randn(shape, device=device)
        viewed_tensors = utils._convert_to_SO3part_view(tensor, cell_index)
        assert(torch.allclose(tensor, viewed_tensors))

    @pytest.mark.parametrize('shapes', [((6, 5, 2, 32), 2),
                                        ((6, 5, 2, 30), 2),
                                        ((5, 3, 33), 1),
                                        ])
    def test_init_strides_after_view(self, shapes):
        shape, cell_index = shapes
        output = utils._initialize_in_SO3part_view(shape, torch.zeros, cell_index)
        viewed_output = utils._convert_to_SO3part_view(output, cell_index)
        assert(viewed_output.stride() == output.stride())
        assert(output.data_ptr() == viewed_output.data_ptr())
