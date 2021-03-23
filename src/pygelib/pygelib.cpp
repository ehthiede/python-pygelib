#include <torch/torch.h>
/* #include <torch/extension.h> */

/* #include "Cnine_base.cpp" */
/* #include "CtensorObj_funs.hpp" */
/* #include "CtensorPackObj_funs.hpp" */
/* #include "CnineSession.hpp" */

#include "GElib_base.cpp"
/* #include "SO3vecObj.hpp" */
/* #include "SO3vecObj_funs.hpp" */
#include "SO3vec.hpp"
#include "SO3element.hpp"
#include "SO3partA.hpp"
#include "SO3part.hpp"
#include "CtensorObj.hpp"
#include "CtensorA.hpp"
#include "CscalarObj.hpp"

/* #include <pybind11/stl_bind.h> */

using namespace cnine;
using namespace GElib;

typedef CtensorObj Ctensor;

#include "_SO3part.hpp"
#include "_Ctensor.hpp"


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

  // Generic Tensors
  pybind11::class_<CscalarObj>(m,"_Cscalar")
    .def(pybind11::init<const float&>())
    .def(pybind11::init<const complex<float>&>())
    .def("str",&CscalarObj::str)
    ;

  pybind11::class_<CtensorObj>(m,"_Ctensor")
    .def(pybind11::init<vector<int>&, const fill_zero&>())
    .def(pybind11::init<vector<int>&, const fill_sequential&>())
    .def(pybind11::init<vector<int>&, const fill_gaussian&>())
    .def("add_mprod", &CtensorObj::add_mprod)
    .def("transp", &CtensorObj::transp)
    /* .def("add", &CtensorObj::add) */
    .def("subtract", &CtensorObj::subtract)
    ;

  /* pybind11::class_<CtensorPackObj>(m,"CtensorPack") */
  /*   .def(pybind11::init<vector<vector<int>>&, const fill_zero&>()); */

  // SO3 Equivariant tensors
  pybind11::class_<SO3type>(m,"_SO3type")
    .def(pybind11::init<vector<int> >())
    .def(pybind11::init<initializer_list<int> >())
    .def("str",&SO3type::str);

  pybind11::class_<SO3element>(m,"_SO3element")
    .def(pybind11::init<const double&, const double&, const double&>())
    ;

  pybind11::class_<SO3vec>(m,"_SO3vec")
    .def(pybind11::init<const SO3type&, const fill_zero&>())
    .def(pybind11::init<const SO3type&, const fill_gaussian&>())
    .def(pybind11::init<const SO3type&, const fill_sequential&>())
    .def("str",&SO3vec::str)
    .def("add_CGproduct",&SO3vec::add_CGproduct)
    .def("add_CGproduct_back0",&SO3vec::add_CGproduct_back0)
    .def("add_CGproduct_back1",&SO3vec::add_CGproduct_back1)
    /* .def("subtract",&SO3vec::subtract) */
    /* .def("add",&SO3vec::add) */
    ;

  pybind11::class_<SO3part>(m,"_SO3part")
    .def(pybind11::init<int, int, const fill_zero&>())
    .def(pybind11::init<int, int, const fill_gaussian&>())
    .def(pybind11::init<int, int, const fill_sequential&>())
    .def("str",&SO3part::str)
    .def("spharm",&SO3part::spharm)
    .def("inp",&SO3part::inp)
    /* .def("CGproduct",&SO3part::CGproduct) */
    /* .def("rotate",&SO3part::rotate) */
    /* .def("get",&SO3part::get) */
    /* .def("set",&SO3part::set_value) */
    /* .def("add", &SO3part::add_temp); */
    ;

  /* pybind11::class_<SO3vec>(m,"_SO3vec") */
  /*   .def(pybind11::init<const SO3type&, const fill_zero&>()) */
  /*   .def(pybind11::init<const SO3type&, const fill_gaussian&>()) */
  /*   .def(pybind11::init<const SO3type&, const fill_sequential&>()) */
  /*   .def("str",&SO3vec::str); */

  // Functions on SO3parts
  m.def("_construct_SO3part_from_Tensor", &SO3partFromTensor, "Constructs an SO3part from a pytorch Tensor");

  // Functions on Ctensors
  m.def("_construct_Ctensor_from_Tensor", &CtensorFromTensor, "Constructs a Ctensor from a pytorch Tensor");
  m.def("iadd_ctensor", &iadd_ctensor, "plus equals operation for ctensor");
  m.def("add_ctensor", &add_ctensor, "sum operation for ctensor");
  m.def("isubtract_ctensor", &isubtract_ctensor, "minus equals operation for ctensor");
  m.def("subtract_ctensor", &subtract_ctensor, "minus operation for ctensor");

  // Functions on SO3vecs
  /* m.def("_SO3vec_times_scalar", &Generic_times_scalar_expr, "So3 vector multiplication w. scalar"); */
  /* m.def("_SO3vec_times_scalar", &Generic_times_scalar_expr, "So3 vector multiplication w. scalar"); */
}
