import torch
import pygelib_cpp as pcpp

a = pcpp.GElibSession()
# pcpp.init_parts_and_dont_take_product()
pcpp.init_parts_and_take_product()
print("Finished GElib Routines")
output_tensor = torch.zeros(2, device='cuda')
print(output_tensor)
print(a)

