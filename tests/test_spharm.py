import pytest
import numpy as np
import torch
from pygelib.spharm import pos_to_rep, rep_to_pos
from pygelib.rotations import EulerRot, WignerD
from pygelib.utils import move_from_end, move_to_end


class TestSpharmConversions:
    @pytest.mark.parametrize('shapes', [((6, 5, 3, 32), -2),
                                        ((6, 5, 5, 3), -1),
                                        ((5, 3, 33), -2)
                                        ])
    @pytest.mark.parametrize('device', [torch.device('cuda'), torch.device('cpu')])
    def test_pos_to_rep_equivariance(self, shapes, device):
        tnsr_shape, xyzdim = shapes
        print(tnsr_shape)
        # xyzdim = len(tnsr_shape) + xyzdim

        # Init x and its rotated version.
        x = torch.randn(tnsr_shape, device=device)
        print(x.shape, "initial")

        alpha, beta, gamma = tuple(np.random.randn(3))
        rot = EulerRot(alpha, beta, gamma).to(device)
        x_rot = move_to_end(torch.clone(x), xyzdim)
        x_rot = x_rot @ rot
        x_rot = move_from_end(x_rot, xyzdim)

        x_spharm = pos_to_rep(x, xyzdim)
        print(x_spharm.shape)
        x_rot_spharm = pos_to_rep(x_rot, xyzdim)

        D = WignerD(1, alpha, beta, gamma, device=device)
        print(x_spharm.shape, D.shape, xyzdim)
        x_spharm = move_to_end(x_spharm, xyzdim)
        print(x_spharm.shape)
        x_spharm_rot_r = x_spharm[0] @ D[0] - x_spharm[1] @ D[1]
        x_spharm_rot_i = x_spharm[1] @ D[0] + x_spharm[0] @ D[1]
        x_spharm_rot = torch.stack([x_spharm_rot_r, x_spharm_rot_i], dim=0)
        x_spharm_rot = move_from_end(x_spharm_rot, xyzdim)

        assert(torch.allclose(x_spharm_rot, x_rot_spharm, atol=1e-6))

    def test_rep_to_pos_equivariance(self):
        return

    def test_inverse(self):
        return
