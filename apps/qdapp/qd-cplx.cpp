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
    Tensor<std::complex<qd_real>> A({3},   csr);
    Tensor<std::complex<qd_real>> B({3,2}, csf);
    Tensor<std::complex<qd_real>> c({2},     sv);

    qd_real a("1.00210000000000002100000000000009887381000000000001");
    qd_real b("4.00000320000000000560000000000012987800000000000000000001");
    qd_real d("2.00210000000000003100000000000098378947000000000000001");
    qd_real e("3.0004200000000002200000000009893809240000000000000001");
    qd_real f("4.0000032000000098837470878000000000020837400000000001");


    // Insert data into B and c
    B.insert({0,0}, std::complex<qd_real>(a,b));
    B.insert({1,0}, std::complex<qd_real>(a-b,b+d));
    B.insert({2,1}, std::complex<qd_real>(a*e,d-f));
    c.insert({0}, std::complex<qd_real>(f+a,d*b));
    c.insert({1}, std::complex<qd_real>(e*f,b+d));

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
    // Exact results:
    // -27.0210436370604920591037214712184606796832136594219 + 28.033652866904391950704162956127059468371975506608 I
    // -63.0631029106848842251038076945568036277866583274359 + 6.014712429843909846368561284913223151507995967952 I
    // 48.077351933653654489269071953418714639165626103059 + 5.931565032573056945704838229391438511903102549051 I

    fpu_fix_end(&oldcw);
    return 0;
}
