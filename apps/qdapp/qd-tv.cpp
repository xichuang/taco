#include <iostream>
#include <iomanip>
#include "taco.h"
#include "qd/fpu.h"
#include "qd/qd_real.h"

using namespace taco;

int main(int argc, char* argv[]) {
    unsigned int oldcw;
    fpu_fix_start(&oldcw);
    // Create formats
  Format csr({Dense});
  Format csf({Sparse,Sparse});
  Format  sv({Sparse});

  // Create tensors
  Tensor<qd_real> A({3},   csr);
  Tensor<qd_real> B({3,2}, csf);
  Tensor<qd_real> c({2},     sv);

  // Insert data into B and c
  B.insert({0,0}, qd_real("1.00210000000000002100000000000000000000000000001"));
  B.insert({1,0}, qd_real("2.00210000000000003100000000000000000000000000001"));
  B.insert({2,1}, qd_real("3.00042000000000022000000000000000000000000000001"));
  c.insert({0}, qd_real("4.00000320000000000560000000000000000000000000000001"));
  c.insert({1}, qd_real("5.00000430000000000003300000000000000000000000000001"));

  // Pack data as described by the formats
  B.pack();
  c.pack();

  // Form a tensor-vector multiplication expression
  IndexVar i, j, k;
  A(i) = B(i,k) * c(k);

  // Compile the expression
  A.compile();

  // Assemble A's indices and numerically compute the result
  A.assemble();
  A.compute();

  std::cout <<std::setprecision(64)<< A << std::endl;

    fpu_fix_end(&oldcw);
    return 0;
}
