This fork aims to add support for double-double and
quad-double scalar type through qd.
Changes to the original code is listed as follows:

1.add qd path to TACO_INCLUDE_DIR and TACO_LIBRARY_DIR in CMakeLists.txt
2.include "qd/dd_real.h" and "qd/qd_real.h" in include/type.h and codegen_c.cpp
3. Use DDReal and DDComplex in Datatype::kind in type.h 
and to return type of Float(), Complex() and type().
4. Use g++ as compiler, set file_ending as "cpp" in module.cpp. add extern "C" arguments to CodeGen_C::generateShim in codegen_c.h.
5. remove keyword "restrict" in codegen_c.h, ir_printer.cpp and codegen.cpp
6. add utils in ir.cpp, array.cpp and typed_value.cpp