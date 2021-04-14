#include <torch/torch.h>

#include "GElib_base.cpp"
#include "SO3partArray.hpp"
#include "GElibSession.hpp"

using namespace cnine;
using namespace GElib;
typedef CtensorObj Ctensor;

/* #include "_SO3part.hpp" */
/* #include "_Ctensor.hpp" */
#include "_SO3partArraytemp.hpp"
#include "SO3partArrayA.hpp"
#include "SO3partArray.hpp"


//CGpowerOp CGpowerOpPy_init(vector< vector<at::Tensor> > x, const int k){
//  return CGpowerOp(SO3vector(x),k);
//}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<fill_zero>(m,"_fill_zero")
    .def(pybind11::init<>());
  pybind11::class_<fill_gaussian>(m,"_fill_gaussian")
    .def(pybind11::init<>());
  pybind11::class_<fill_sequential>(m,"_fill_sequential")
    .def(pybind11::init<>());

  //////////// START
  pybind11::class_<SO3partArray>(m,"_SO3partArray")
    /* .def(pybind11::init<vector<int>&, int, int, int>()) */
    .def(pybind11::init<vector<int>&, int, int, fill_gaussian&, int>())
    .def("str", &SO3partArray::str)
    /* .def(pybind11::init<vector<int>&, int, int, fill_gaussian&, int>()); */
    /* .def("CGproduct", &SO3part::CGproduct) */
    ;

  pybind11::class_<GElibSession>(m,"GElibSession")
    .def(pybind11::init<>());

  m.def("sampleprint", &sampleprint, "Construct and print an SO3partArray");
  m.def("add_SO3partArrays", &add_SO3partArrays, "Construct and print an SO3partArray");
  m.def("partArrayCGproduct", &partArrayCGproduct, "Construct and print an SO3partArray");
}
