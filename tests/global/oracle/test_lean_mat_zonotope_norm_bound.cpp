// test_lean_mat_zonotope_norm_bound - Zonotope::norm against the CORALean oracle, as a bound
//
// The norm is that of the enclosing interval matrix. The norms one and inf are sums and maxima of
// the same entries: bit for bit on the lean backend. The Frobenius norm is a square root, which the
// oracle bounds from above by Newton steps: CORA.cpp's correctly rounded root is inside the bound
// and within kRelative of it, but not its bits, so there is no ORACLE tag (as for the interval
// matrix).
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/mat_exp/mat_exp.h"
#include "cora/global/oracle/mat_zonotope.h"
#include "oracle_testing.h"

#include <limits>

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The norm of a set through ContSet::norm, which takes the type of a matrix norm.
Tensor set_norm(const ContSet &S, Tensor::Norm type) { return S.norm(type); }

constexpr double kRelative = 4 * std::numeric_limits<double>::epsilon();

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("matZonotope.norm", [](test::oracle::Random &rnd) {
        for (const auto &[r, c, h] : {std::array<int64_t, 3>{2, 2, 1}, {3, 4, 2}, {5, 1, 3}}) {
            const Zonotope A = rnd.mat_zonotope(r, c, h);
            const lean::MatZonotope L(A);
            test::oracle::check(set_norm(A, Tensor::Norm::one), L.norm("1"));
            test::oracle::check(set_norm(A, Tensor::Norm::inf), L.norm("inf"));
            const double root = set_norm(A, Tensor::Norm::fro).data()[0], bound = L.norm("fro");
            test::check(root <= bound && root >= bound * (1 - kRelative),
                        "matZonotope.norm fro: below the bound and close to it");
        }
    });
    return test::finish("lean mat_zonotope norm bound");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
