import torch
from pygelib.SO3VecArray import SO3VecArray
from time import time


def _calculate_mean(tensor_list):
    mean_0 = torch.mean(tensor_list[0])
    for p in tensor_list[1:]:
        mean_0 += torch.mean(p)
    return mean_0


def profile_cg_product(cg_module, n_atoms, l_As, l_Bs, n_channels_A, n_channels_B, device, n_reps=100):
    shape_As = [(2, n_atoms, 2 * l + 1, n_channels_A) for l in l_As]
    shape_Bs = [(2, n_atoms, 2 * l + 1, n_channels_B) for l in l_Bs]
    times = []
    for i in range(n_reps):
        start_time = time()
        A = SO3VecArray([torch.randn(s_i, device=device, requires_grad=True) for s_i in shape_As])
        B = SO3VecArray([torch.randn(s_i, device=device, requires_grad=True) for s_i in shape_Bs])
        prod = cg_module(A, B)
        _calculate_mean(prod).backward()
        times.append(time() - start_time)
    return times


def check_module_output_agrees(cg_module1, cg_module2, n_atoms, l_As, l_Bs, n_channels_A, n_channels_B, device):
    shape_As = [(2, n_atoms, 2 * l + 1, n_channels_A) for l in l_As]
    shape_Bs = [(2, n_atoms, 2 * l + 1, n_channels_B) for l in l_Bs]
    
    A = SO3VecArray([torch.randn(s_i, device=device, requires_grad=True) for s_i in shape_As])
    B = SO3VecArray([torch.randn(s_i, device=device, requires_grad=True) for s_i in shape_Bs])
    prod1 = cg_module1(A, B)
    prod2 = cg_module2(A, B)

    for i, j in zip(prod1, prod2):
        assert(torch.allclose(i, j, atol=5e-6))
