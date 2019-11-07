set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(progs dd-tv qd-tv dd-cplx tenv-cplx qd-cplx)
foreach(prog ${progs})
  add_executable(${prog} "${prog}.cpp" )
  target_link_libraries(${prog} LINK_PUBLIC taco qd)
endforeach()
