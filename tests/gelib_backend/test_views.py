import pytest
import torch
from pygelib.utils import _convert_to_SO3part_view


@pytest.mark.parametrize('shapes', [((3, 5, 31), 1),
                                    ((4, 32), 1),
                                    ((2, 2, 33,), 2)])
@pytest.mark.parametrize('device', [torch.device('cuda'), torch.device('cpu')])
def test_views(shapes, device):
    shape, cell_index = shapes
    tensor = torch.randn(shape, device=device)
    viewed_tensors = _convert_to_SO3part_view(tensor, cell_index)
    assert(torch.allclose(tensor, viewed_tensors))
