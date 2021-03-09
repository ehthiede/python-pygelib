#include <torch/torch.h>

#include "GElib_base.cpp"
#include "SO3vecObj_funs.hpp"

using namespace cnine;
using namespace GElib;


//CGpowerOp CGpowerOpPy_init(vector< vector<at::Tensor> > x, const int k){
//  return CGpowerOp(SO3vector(x),k);
//}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<SO3type>(m,"SO3type")
    .def(pybind11::init<vector<int> >())
    .def(pybind11::init<initializer_list<int> >())
    .def("str",&SO3type::str);
}

/* PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { */
/*   m.def("forward", &lltm_forward, "LLTM forward"); */
/*   m.def("backward", &lltm_backward, "LLTM backward"); */
/* } */
