#include <iostream>
#include <iomanip>
#include "taco.h"
#include "qd/fpu.h"
#include "qd/dd_real.h"

using namespace taco;

int main(int argc, char* argv[]) {
    unsigned int oldcw;
    fpu_fix_start(&oldcw);
    // Create formats
    Format csr({Dense});
    Format csf({Sparse,Sparse});
    Format  sv({Sparse});

    // Create tensors
    Tensor<std::complex<double>> A({3},   csr);
    Tensor<std::complex<double>> B({3,2}, csf);
    Tensor<std::complex<double>> c({2},     sv);

    std::complex<double> a(0,1.00210000000000002100000000000000000000000000001);
    std::complex<double> b(0,4.00000320000000000560000000000000000000000000000001);

    // Insert data into B and c
    B.insert({0,0}, a);
    B.insert({1,0}, std::complex<double>(2.00210000000000003100000000000000000000000000001));
    B.insert({2,1}, std::complex<double>(3.00042000000000022000000000000000000000000000001));
    c.insert({0}, b);
    c.insert({1}, std::complex<double>(5.00000430000000000003300000000000000000000000000001));

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

    std::cout <<std::setprecision(16)<< A << std::endl;
    // Exact results:
    // 4.0084032067200000896118272000000001176000000000
    // 8.0084064067200001352118592000000001736000000000
    // 15.0021129018060011000999598600000000072600000001

    fpu_fix_end(&oldcw);
    return 0;
}
