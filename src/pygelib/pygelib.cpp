#include <torch/torch.h>
/* #include <torch/extension.h> */

/* #include "Cnine_base.cpp" */
/* #include "CtensorObj_funs.hpp" */
/* #include "CtensorPackObj_funs.hpp" */
/* #include "CnineSession.hpp" */

#include "GElib_base.cpp"
#include "SO3vec.hpp"
#include "SO3element.hpp"
#include "SO3partA.hpp"
#include "SO3part.hpp"
#include "CtensorObj.hpp"
#include "CtensorA.hpp"
#include "CscalarObj.hpp"
#include "SO3partArrayA.hpp"
#include "SO3partArray.hpp"

using namespace cnine;
using namespace GElib;
typedef CtensorObj Ctensor;

/* #include "_SO3part.hpp" */
/* #include "_Ctensor.hpp" */
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

  /* // Generic Tensors */
  /* pybind11::class_<CscalarObj>(m,"_Cscalar") */
  /*   .def(pybind11::init<const float&>()) */
  /*   .def(pybind11::init<const complex<float>&>()) */
  /*   .def("str",&CscalarObj::str) */
  /*   ; */


  //////////// START
  pybind11::class_<CtensorObj>(m,"_Ctensor")
    .def(pybind11::init<vector<int>&, const fill_zero&>())
    .def(pybind11::init<vector<int>&, const fill_sequential&>())
    .def(pybind11::init<vector<int>&, const fill_gaussian&>())
    .def("add_mprod", &CtensorObj::add_mprod)
    .def("transp", &CtensorObj::transp)
    /* .def("add", &CtensorObj::add) */
    .def("subtract", &CtensorObj::subtract)
    ;
  //////////// STOP

  /* /1* pybind11::class_<CtensorPackObj>(m,"CtensorPack") *1/ */
  /* /1*   .def(pybind11::init<vector<vector<int>>&, const fill_zero&>()); *1/ */

  /* // SO3 Equivariant tensors */
  /* pybind11::class_<SO3type>(m,"_SO3type") */
  /*   .def(pybind11::init<vector<int> >()) */
  /*   .def(pybind11::init<initializer_list<int> >()) */
  /*   .def("str",&SO3type::str); */

  /* pybind11::class_<SO3element>(m,"_SO3element") */
  /*   .def(pybind11::init<const double&, const double&, const double&>()) */
  /*   ; */

  /* pybind11::class_<SO3vec>(m,"_SO3vec") */
  /*   .def(pybind11::init<const SO3type&, const fill_zero&>()) */
  /*   .def(pybind11::init<const SO3type&, const fill_gaussian&>()) */
  /*   .def(pybind11::init<const SO3type&, const fill_sequential&>()) */
  /*   .def("str",&SO3vec::str) */
  /*   .def("add_CGproduct",&SO3vec::add_CGproduct) */
  /*   .def("add_CGproduct_back0",&SO3vec::add_CGproduct_back0) */
  /*   .def("add_CGproduct_back1",&SO3vec::add_CGproduct_back1) */
  /*   /1* .def("subtract",&SO3vec::subtract) *1/ */
  /*   /1* .def("add",&SO3vec::add) *1/ */
  /*   ; */

  //////////// START
  pybind11::class_<SO3partArray>(m,"_SO3partArray")
    /* .def(pybind11::init<vector<int>&, int, int, int>()) */
    .def(pybind11::init<vector<int>&, int, int, fill_gaussian&, int>())
    .def("str", &SO3partArray::str)
    /* .def(pybind11::init<vector<int>&, int, int, fill_gaussian&, int>()); */
    /* .def("CGproduct", &SO3part::CGproduct) */
    ;

  m.def("_partArray_CGproduct", &partArrayCGproduct, "Takes cg product of two part arrays");
  /* m.def("_partArray_CGproduct", &py::overload_cast<const SO3partArray&, const SO3partArray&, const int l>(&CGproduct)); */
  //////////// STOP

  /* pybind11::class_<SO3part>(m,"_SO3part") */
  /*   .def(pybind11::init<int, int, const fill_zero&>()) */
  /*   .def(pybind11::init<int, int, const fill_gaussian&>()) */
  /*   .def(pybind11::init<int, int, const fill_sequential&>()) */
  /*   .def("str",&SO3part::str) */
  /*   .def("spharm",&SO3part::spharm) */
  /*   .def("inp",&SO3part::inp) */
  /*   /1* .def("CGproduct",&SO3part::CGproduct) *1/ */
  /*   /1* .def("rotate",&SO3part::rotate) *1/ */
  /*   /1* .def("get",&SO3part::get) *1/ */
  /*   /1* .def("set",&SO3part::set_value) *1/ */
  /*   /1* .def("add", &SO3part::add_temp); *1/ */
  /*   ; */

  /* /1* pybind11::class_<SO3vec>(m,"_SO3vec") *1/ */
  /* /1*   .def(pybind11::init<const SO3type&, const fill_zero&>()) *1/ */
  /* /1*   .def(pybind11::init<const SO3type&, const fill_gaussian&>()) *1/ */
  /* /1*   .def(pybind11::init<const SO3type&, const fill_sequential&>()) *1/ */
  /* /1*   .def("str",&SO3vec::str); *1/ */

  //////////// START
  // Functions on SO3partArrays
  m.def("_construct_SO3partArray_from_Tensor", &SO3partArrayFromTensor, "Constructs an SO3partArray from a pytorch Tensor");

  m.def("_construct_Tensor_from_SO3partArray", &MoveSO3partArrayToTensor, "Constructs a pytorch Tensor from an SO3partArray");

  m.def("add_SO3partArrays", &add_SO3partArrays, "Adds SO3partArrays");

  m.def("partArrayCGproduct", &partArrayCGproduct, "CGproduct of two SO3partArrays");

  /* // Functions on SO3parts */
  /* m.def("_construct_SO3part_from_Tensor", &SO3partFromTensor, "Constructs an SO3part from a pytorch Tensor"); */

  /* // Functions on Ctensors */
  /* m.def("_construct_Ctensor_from_Tensor", &CtensorFromTensor, "Constructs a Ctensor from a pytorch Tensor"); */
  //////////// STOP


  /* m.def("iadd_ctensor", &iadd_ctensor, "plus equals operation for ctensor"); */
  /* m.def("add_ctensor", &add_ctensor, "sum operation for ctensor"); */
  /* m.def("isubtract_ctensor", &isubtract_ctensor, "minus equals operation for ctensor"); */
  /* m.def("subtract_ctensor", &subtract_ctensor, "minus operation for ctensor"); */

  /* // Functions on SO3vecs */
  /* /1* m.def("_SO3vec_times_scalar", &Generic_times_scalar_expr, "So3 vector multiplication w. scalar"); *1/ */
  /* /1* m.def("_SO3vec_times_scalar", &Generic_times_scalar_expr, "So3 vector multiplication w. scalar"); *1/ */
}
