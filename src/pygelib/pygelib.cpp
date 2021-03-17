#include <torch/torch.h>

/* #include "Cnine_base.cpp" */
/* #include "CtensorObj_funs.hpp" */
/* #include "CtensorPackObj_funs.hpp" */
/* #include "CnineSession.hpp" */

#include "GElib_base.cpp"
#include "SO3vecObj.hpp"
#include "SO3vecObj_funs.hpp"
#include "SO3part.hpp"

/* #include <pybind11/stl_bind.h> */

using namespace cnine;
using namespace GElib;


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
    .def(pybind11::init<vector<int>&, const fill_gaussian&>());

  /* pybind11::class_<CtensorPackObj>(m,"CtensorPack") */
  /*   .def(pybind11::init<vector<vector<int>>&, const fill_zero&>()); */

  // SO3 Equivariant tensors
  pybind11::class_<SO3type>(m,"_SO3type")
    .def(pybind11::init<vector<int> >())
    .def(pybind11::init<initializer_list<int> >())
    .def("str",&SO3type::str);

  pybind11::class_<SO3vecObj>(m,"_SO3vec")
    .def(pybind11::init<const SO3type&, const fill_zero&>())
    .def(pybind11::init<const SO3type&, const fill_gaussian&>())
    .def(pybind11::init<const SO3type&, const fill_sequential&>())
    .def("str",&SO3vecObj::str);

  pybind11::class_<SO3partObj>(m,"_SO3part")
    .def(pybind11::init<int, int, const fill_zero&>())
    .def(pybind11::init<int, int, const fill_gaussian&>())
    .def(pybind11::init<int, int, const fill_sequential&>())
    .def("str",&SO3partObj::str)
    .def("get",&SO3partObj::get)
    .def("spharm",&SO3partObj::spharm)
    .def("rotate",&SO3partObj::rotate)
    .def("set",&SO3partObj::set_value)
    .def("add", &SO3partObj::add_temp);

  pybind11::class_<SO3vecObj>(m,"_SO3vec")
    .def(pybind11::init<const SO3type&, const fill_zero&>())
    .def(pybind11::init<const SO3type&, const fill_gaussian&>())
    .def(pybind11::init<const SO3type&, const fill_sequential&>())
    .def("str",&SO3vecObj::str);

  // Functions on SO3parts
  /* m.def("_spharm", &spharm, "Convert Cartesian tensor to spherical harmonics"); */

  // Functions on SO3vecs
  /* m.def("_SO3vec_times_scalar", &Generic_times_scalar_expr, "So3 vector multiplication w. scalar"); */
  /* m.def("_SO3vec_times_scalar", &Generic_times_scalar_expr, "So3 vector multiplication w. scalar"); */
}
