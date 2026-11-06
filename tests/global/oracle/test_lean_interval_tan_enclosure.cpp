// test_lean_interval_tan_enclosure - Interval::tan against the CORALean oracle, as enclosure
//
// The oracle's tan (from batch 5) is an extended box: a coordinate whose range is not certified
// free of the poles k pi + pi/2 is (-Inf, Inf), else `tanDown` at the lower end and `tanUp` at the
// upper one (`Interval.tan`, `tan_outer`). CORA's box is the tangent of the corners where the range
// is narrower than pi and holds no pole, else (-Inf, Inf). So the check is the Lean statement:
// CORA.cpp's box lies inside the oracle's, which is unbounded wherever CORA's is; where the oracle
// is bounded, CORA's is too and the bounds agree to kAgree, relative (libm against the oracle's
// halvings). No bit-exact test exists, so there is no ORACLE tag. The boxes run over several poles.
// The oracles before batch 5 gave inverted bounds over a pole; with them the test is skipped.
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace cora;


namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr int kTerms = 20;
constexpr double kAgree = 1e-9;

/// Whether the oracle's tan is unbounded over the pole pi/2 (it is from batch 5 on).
bool aux_pole_aware() {
    const Tensor lo = Tensor::from_data({1.5}, {1, 1}), hi = Tensor::from_data({1.6}, {1, 1});
    const std::pair<Tensor, Tensor> box = lean::Interval(Interval(lo, hi)).tan(kTerms).gather();
    return std::isinf(box.first.item()) && std::isinf(box.second.item());
}

/// Boxes of n coordinates, each of width at most 3, in [-5, 5]: they meet the poles at +-pi/2 and
/// -3 pi / 2, 3 pi / 2.
Interval aux_boxes(oracle::Random &rnd, int n) {
    std::vector<double> lo(n), hi(n);
    for (int i = 0; i < n; ++i) {
        lo[i] = rnd.uniform(-5, 5);
        hi[i] = lo[i] + rnd.uniform(0, 3);
    }
    return {test::column(lo), test::column(hi)};
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    oracle::run("interval.tan", [](oracle::Random &rnd) {
        if (oracle::mode() == oracle::Mode::exact) return; // nearest gives no enclosure
        if (!aux_pole_aware())
            return oracle::skipped("the oracle's interval.tan holds no pole test (older oracle)");
        const Interval I = aux_boxes(rnd, 200);
        const Interval R = I.tan();
        const lean::Interval O = lean::Interval(I).tan(kTerms);
        oracle::check_enclosed("tan", R, O);
        const auto [olo, ohi] = O.gather();
        const std::vector<double> a = R.inf().data(), b = R.sup().data(), c = olo.data(),
                                  d = ohi.data();
        int bounded = 0, unbounded = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const bool oracle_bounded = std::isfinite(c[i]) && std::isfinite(d[i]);
            if (!oracle_bounded) {
                ++unbounded;
                continue;
            }
            ++bounded;
            test::check(std::isfinite(a[i]) && std::isfinite(b[i]),
                        "tan: the oracle's bound is finite, so CORA's is");
            for (const auto &[x, y] : {std::pair{a[i], c[i]}, {b[i], d[i]}})
                test::check(std::abs(x - y) <= kAgree * std::max(1.0, std::abs(y)),
                            "tan: the bounds agree");
        }
        test::check(bounded > 20 && unbounded > 20, "tan: boxes with and without poles");
        std::cerr << "  tan: " << bounded << " bounded, " << unbounded
                  << " unbounded coordinates\n";
    });
    return test::finish("lean interval tan enclosure");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
