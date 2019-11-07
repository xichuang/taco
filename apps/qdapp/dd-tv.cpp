#include <iostream>
#include <iomanip>
#include "taco.h"
#include "qd/dd_real.h"

using namespace taco;

int main(int argc, char* argv[]) {
  // Create formats
  Format csr({Dense,Sparse});
  Format csf({Sparse,Sparse,Sparse});
  Format  sv({Sparse});

  // Create tensors
  Tensor<qd_real> A({2,3},   csr);
  Tensor<qd_real> B({2,3,4}, csf);
  Tensor<qd_real> c({4},     sv);

  // Insert data into B and c
  B.insert({0,0,0}, dd_real("1.0000000000000000000000000000000000000000000000000000000001"));
  B.insert({1,2,0}, dd_real("2.000000000000000000000000000000000000000000000000000001"));
  B.insert({1,2,1}, dd_real("3.0000000000000000000000000000000000000000000000000000001"));
  c.insert({0}, dd_real("4.000000000000000000000000000000000000000000000000000000001"));
  c.insert({1}, dd_real("5.000000000000000000000000000000000000000000000000000000001"));

  // Pack data as described by the formats
  B.pack();
  c.pack();

  // Form a tensor-vector multiplication expression
  IndexVar i, j, k;
  A(i,j) = B(i,j,k) * c(k);

  // Compile the expression
  A.compile();

  // Assemble A's indices and numerically compute the result
  A.assemble();
  A.compute();

  std::cout <<std::setprecision(32)<< A << std::endl;
}
