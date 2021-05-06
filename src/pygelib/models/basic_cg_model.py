import torch


class Nodewise(torch.nn.Module):
    def __init__(self, params, basis_fxn):
        super(EdgeFeaturizer, self).__init__()
        self.bf_params = params

    def forward(self, x):


