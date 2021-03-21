import torch
import pygelib_cpp

class Ctensor(Ctensor):
    def __init__(self, real_tensor, imag_tensor=None):
        if imag_tensor is None:
            imag_tensor = torch.zeros_like(real_tensor)

        so3type = pygelib_cpp._SO3type(typelist)
        self.data = pygelib_cpp._SO3vec(so3type, fill)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def __setitem__(self, idx, part):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError

    def __add__(self, other):
        raise NotImplementedError

    __radd__ = __add__

    def __sub__(self, other):
        raise NotImplementedError

    __rsub__ = __sub__

    def __mul__(self, other):
        raise NotImplementedError

    def __abs__(self, other):
        raise NotImplementedError

    @property
    def grad(self):
        raise NotImplementedError

    @classmethod
    def requires_grad(cls):
        raise NotImplementedError

    def requires_grad_(self, requires_grad=True):
        raise NotImplementedError

import pygelib_cpp

class Ctensor(Ctensor):
