// test_lean_ellipsoid_bound - Ellipsoid::linear_map against the CORALean oracle, within ulps
//
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "cora/global/oracle/ellipsoid.h"
#include "oracle_testing.h"

#include <algorithm>
#include <cmath>
#include <limits>

using namespace cora;
namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// How far CORA.cpp may be from the oracle's M Q M' and M q, in units in the last place.
constexpr double kMtimesUlps = 64;

/// Whether the entries of a and b differ by at most `ulps` units in the last place of the larger.
bool aux_near(const std::vector<double> &a, const std::vector<double> &b, double ulps) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double larger = std::max(std::abs(a[i]), std::abs(b[i]));
        const double unit =
            std::nextafter(larger, std::numeric_limits<double>::infinity()) - larger;
        if (std::abs(a[i] - b[i]) > ulps * unit) return false;
    }
    return true;
}

/// A random positive definite ellipsoid of n dimensions around the origin when `centered`.
Ellipsoid aux_ellipsoid(oracle::Random &rnd, int64_t n, bool centered) {
    const Tensor A = rnd.matrix(n, n);
    const Tensor Q = A.matmul(A.matrix_transpose()) + A.eye_like(n) * 0.5;
    return {(Q + Q.matrix_transpose()) * 0.5, centered ? Q.zeros_like({n, 1}) : rnd.vector(n)};
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    // CORALean inflates M Q M' upward and does not symmetrize: the entries agree to the stated ulps
    oracle::run("ellipsoid.mtimes", [](oracle::Random &rnd) {
        const Ellipsoid E = aux_ellipsoid(rnd, 3, false);
        const Tensor M = rnd.matrix(2, 3);
        const Ellipsoid theirs = lean::Ellipsoid(E).linear_map(M).nominal();
        const Ellipsoid mine = E.linear_map(M);
        test::check(aux_near(mine.Q().data(), theirs.Q().data(), kMtimesUlps),
                    "ellipsoid.mtimes: Q within 64 ulp");
        test::check(aux_near(mine.q().data(), theirs.q().data(), kMtimesUlps),
                    "ellipsoid.mtimes: q within 64 ulp");
    });
    return test::finish("lean ellipsoid bounds");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
