// test_lean_zonotope_zonotope_norm_bound - Zonotope::zonotope_norm against the CORALean oracle, as
// a bound
//
// The oracle's value (from batch 5) is a lower bound on min {|g|_inf | G g = p} (`zonotopeNorm_le`):
// the norm itself, up to the rounding of its quotients, for n independent generators, and infinite
// only where it certifies that p is outside the span of the generators. CORA.cpp's norm is that
// minimum from a linear program. So the oracle's value never exceeds CORA's beyond kAgree
// (relative), is the norm to kAgree for n independent generators, and is infinite only where CORA's
// is. There is no ORACLE tag: no bit-exact test exists.

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/oracle/zonotope.h"
#include "oracle_testing.h"

#include <algorithm>
#include <cmath>

using namespace cora;
namespace oracle = test::oracle;

namespace {

constexpr double kAgree = 1e-9;

/// The oracle's lower bound against CORA.cpp's norm, which is `independent` for n generators.
void aux_check(const Zonotope &Z, const Tensor &p, bool independent) {
    const double norm = Z.zonotope_norm(p).item();
    const double bound = lean::Zonotope(Z).zonotope_norm(lean::Tensor::from(p));
    const double slack = kAgree * std::max(1.0, std::abs(norm));
    if (std::isinf(bound))
        return test::check(std::isinf(norm), "zonotopeNorm: infinite only where CORA's is");
    test::check(bound <= norm + slack, "zonotopeNorm: a lower bound");
    if (independent) test::check(std::abs(bound - norm) <= slack, "zonotopeNorm: the norm for n generators");
}

} // namespace

int main() {
    if (!oracle::available()) return 0;
    oracle::run("zonotope.zonotopeNorm", [](oracle::Random &rnd) {
        for (const int n : {1, 2, 3}) {
            // n independent generators
            const Zonotope square(Tensor::zeros({n, 1}), rnd.matrix(n, n));
            for (int k = 0; k < 3; ++k) aux_check(square, rnd.vector(n), true);
            // more generators than dimensions: the oracle bounds the minimum from below
            const Zonotope wide(Tensor::zeros({n, 1}), rnd.matrix(n, n + 2));
            for (int k = 0; k < 3; ++k) aux_check(wide, rnd.vector(n), false);
        }
        // generators in a plane (the last coordinate 0): a point off the plane has no norm
        for (const int n : {2, 3}) {
            const Tensor G = Tensor::concat({rnd.matrix(n - 1, n + 1), Tensor::zeros({1, n + 1})}, -2);
            const Zonotope flat(Tensor::zeros({n, 1}), G);
            aux_check(flat, rnd.vector(n), false);
            aux_check(flat, Tensor::concat({rnd.vector(n - 1), Tensor::zeros({1, 1})}, -2), false);
        }
    });
    return test::finish("lean zonotope zonotope_norm bound");
}
