set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(progs dd-tv qd-tv)
foreach(prog ${progs})
  add_executable(${prog} "${prog}.cpp" )
  target_link_libraries(${prog} LINK_PUBLIC taco qd)
endforeach()
