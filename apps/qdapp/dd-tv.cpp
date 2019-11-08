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
    Tensor<dd_real> A({3},   csr);
    Tensor<dd_real> B({3,2}, csf);
    Tensor<dd_real> c({2},     sv);

    double t1=479876442.000000000;
    std::cout <<std::setprecision(32)<< t1 <<"  "<<dd_real(t1)<< std::endl;
    double t2=3.0000001;
    std::cout <<std::setprecision(32)<< t2 <<"  "<<1*t2<<"  "<<dd_real(t2)<< std::endl;


    dd_real a="1.00210000000000002100000000000000000000000000001";
    dd_real b="4.00000320000000000560000000000000000000000000000001";

    // Insert data into B and c
    B.insert({0,0}, a);
    B.insert({1,0}, dd_real("2.00210000000000003100000000000000000000000000001"));
    B.insert({2,1}, dd_real("3.00042000000000022000000000000000000000000000001"));
    c.insert({0}, b);
    c.insert({1}, dd_real("5.00000430000000000003300000000000000000000000000001"));

    std::cout<<std::setprecision(32)<<a*b<<std::endl;

    // Pack data as described by the formats
    B.pack();
    c.pack();
    std::cout <<std::setprecision(32)<< B << std::endl;
    std::cout <<std::setprecision(32)<< c << std::endl;

    // Form a tensor-vector multiplication expression
    IndexVar i, j, k;
    A(i) = B(i,k) * c(k);

    // Compile the expression
    A.compile();

    // Assemble A's indices and numerically compute the result
    A.assemble();
    A.compute();

    std::cout <<std::setprecision(32)<< A << std::endl;
    // Exact results:
    // 4.0084032067200000896118272000000001176000000000
    // 8.0084064067200001352118592000000001736000000000
    // 15.0021129018060011000999598600000000072600000001

    fpu_fix_end(&oldcw);
    return 0;
}
