// test_lean_mat_zonotope_norm_bound - MatZonotope::norm against the CORALean oracle, as a bound
//
// The norm is that of the enclosing interval matrix. The norms one and inf are sums and maxima of the
// same entries: bit for bit on the lean backend. The Frobenius norm is a square root, which the oracle
// bounds from above by Newton steps: CORA.cpp's correctly rounded root is inside the bound and within
// kRelative of it, but not its bits, so there is no ORACLE tag (as for the interval matrix).

#include "cora/global/oracle/mat_zonotope.h"
#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"
#include "oracle_testing.h"

#include <limits>

using namespace cora;

namespace {

constexpr double kRelative = 4 * std::numeric_limits<double>::epsilon();

}

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("matZonotope.norm", [](test::oracle::Random &rnd) {
        for (const auto [r, c, h] : {std::array<int64_t, 3>{2, 2, 1}, {3, 4, 2}, {5, 1, 3}}) {
            const MatZonotope A = rnd.mat_zonotope(r, c, h);
            const lean::MatZonotope L(A);
            test::oracle::check(A.norm(Tensor::Norm::one), L.norm("1"));
            test::oracle::check(A.norm(Tensor::Norm::inf), L.norm("inf"));
            const double root = A.norm(Tensor::Norm::fro).data()[0], bound = L.norm("fro");
            test::oracle::check_enclosed("matZonotope.norm fro", root, bound);
            test::check(root >= bound * (1 - kRelative), "matZonotope.norm fro: close to the bound");
        }
    });
    return test::finish("lean mat_zonotope norm bound");
}
