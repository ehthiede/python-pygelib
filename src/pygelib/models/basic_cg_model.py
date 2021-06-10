import torch
from pygelib_cpp import  estimate_num_products


class CGMPBlock(torch.nn.Module):
    def __init__(self, transformation_dict, real_edges=True):
        super(EdgeFeaturizer, self).__init__()
        
        # construct linear transformation_sizes
        max_input_l = max([transformation_dict.keys()])
        input_ls = np.zeros(max_input_l + 1, dtype=int)
        
        num_products = estimate_num_products(
        




    def forward(self, x):


