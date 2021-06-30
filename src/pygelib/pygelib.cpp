#include <torch/torch.h>
#include <math.h>
#include "GElib_base.cpp"
#include "SO3partArray.hpp"
#include "SO3vec.hpp"
#include "GElibSession.hpp"

using namespace cnine;
using namespace GElib;
typedef CtensorObj Ctensor;

// GElib dependencies
/* #include "_SO3part.hpp" */
/* #include "_Ctensor.hpp" */
#include "SO3partArrayA.hpp"
#include "SO3partArray.hpp"
#include "SO3element.hpp"
#include "SO3partA_CGproduct_back0_cop.hpp"
#include "SO3partA_CGproduct_back1_cop.hpp"

// Python Interface Codes
#include "_SO3partArray.hpp"


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
    /* .def("CGproduct_back0", &CGproduct_back0) */
    /* .def("CGproduct_back1", &CGproduct_back1) */
    /* .def("CGproduct", &CGproduct<SO3partArray&, SO3partArray&, int>) */
    /* .def(pybind11::init<vector<int>&, int, int, fill_gaussian&, int>()); */
    /* .def("CGproduct", &SO3part::CGproduct) */
    ;

  pybind11::class_<GElibSession>(m,"GElibSession")
    .def(pybind11::init<>());

  /* m.def("_internal_SO3partArray_from_Tensor", &SO3partArrayFromTensor, "Constructs a GElib SO3partArray from a pytorch Tensor. Does NOT account for padding."); */
  m.def("_internal_SO3partArray_from_Tensor", &SO3partArrayFromTensor, py::return_value_policy::move);
  /* m.def("_internal_Tensor_from_SO3partArray", &MoveSO3partArrayToTensor, "Constructs a tensor from a GElib SO3partArray. Does NOT account for padding."); */
  m.def("_sampleprint_test", &sampleprint, "Testing method that constructs and prints an SO3partArray of ones on CPU and GPU.");
  m.def("sum_SO3partArrays_inplace", &sum_SO3partArrays_inplace, "Sum two SO3 part arrays in place");
  m.def("partArrayCGproduct", &partArrayCGproduct, "Performs a CG product of two parts");
  m.def("add_in_partArrayCGproduct", &add_in_partArrayCGproduct, "Construct and print an SO3partArray");
  m.def("add_in_partArrayCGproduct_back0", &add_in_partArrayCGproduct_back0, "Construct and print an SO3partArray");
  m.def("add_in_partArrayCGproduct_back1", &add_in_partArrayCGproduct_back1, "Construct and print an SO3partArray");
  m.def("test_partArrayCGproduct_back0", &test_partArrayCGproduct_back0, "Construct and print an SO3partArray");
  /* m.def("rotate_SO3partArray", &rotate_SO3partArray, "Rotate an SO3partArray"); */

  /* m.def("add_in_partArrayCGproduct_back0", &add_in_partArrayCGproduct_back0, "Construct and print an SO3partArray"); */
  /* m.def("add_in_partArrayCGproduct_back1", &add_in_partArrayCGproduct_back1, "Construct and print an SO3partArray"); */

  // Some testing routines
  /* m.def("test_conversion", &test_conversion, "sum tensors in place"); */
  m.def("TestGelibPtrs", &TestGelibPtrs);
  m.def("get_num_channels", &get_num_channels);
  m.def("estimate_num_products", &estimate_num_products);
}
