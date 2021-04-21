#include <torch/torch.h>
#include <math.h>
#include "GElib_base.cpp"
#include "SO3partArray.hpp"
#include "GElibSession.hpp"

using namespace cnine;
using namespace GElib;
typedef CtensorObj Ctensor;

#include "test_routines.hpp"


//CGpowerOp CGpowerOpPy_init(vector< vector<at::Tensor> > x, const int k){
//  return CGpowerOp(SO3vector(x),k);
//}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  pybind11::class_<GElibSession>(m,"GElibSession")
    .def(pybind11::init<>());

  // Some testing routines
  m.def("init_parts_and_take_product", &init_parts_and_take_product);
  m.def("init_parts_and_dont_take_product", &init_parts_and_dont_take_product);
}
