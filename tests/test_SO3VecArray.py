import torch
import pytest
from pygelib.SO3VecArray import SO3VecArray


class TestSO3VecArray(object):
    @pytest.mark.parametrize('shapes', [(2, 6, 5, 1, 32),
                                        (2, 6, 5, 5, 30),
                                        (2, 5, 3, 33),
                                        ])
    def test_adims(self, shapes):
        a = SO3VecArray([torch.randn(shapes)])
        correct_adims = shapes[1:-2]

        assert(a.adims[0] == correct_adims)
