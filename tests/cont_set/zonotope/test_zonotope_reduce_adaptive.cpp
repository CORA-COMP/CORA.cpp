// test_zonotope_reduce_adaptive - zonotope reduce "adaptive" and "idx": hand cases and enclosure
//
// CORA: contSet/@zonotope/reduce.m
// CORA: contSet/@zonotope/private/priv_reduceAdaptive.m
// PROPERTY: contSet/@zonotope/private/priv_reduceAdaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Two axis generators and one oblique one of Girard metric 0.1: the box is within the allowed
/// distance at 10 % (all are boxed), not at 5 % (only the axis generators are).
void a_hand_case_in_two_dimensions(const std::string &b) {
    const Zonotope Z(Tensor({0.5, -0.5}), Tensor({{1.0, 0.0, 0.1}, {0.0, 1.0, 0.1}}));
    const AdaptiveReduction all = Z.reduce_adaptive(0.1);
    check(test::close(all.Z.generators(), Tensor({{1.1, 0.0}, {0.0, 1.1}}), 1e-12),
          b + ": 10 %: the box");
    check(all.idx == std::vector<int64_t>({0, 1, 2}), b + ": 10 %: every index is boxed");
    check(std::abs(all.error - 0.2) < 1e-12, b + ": 10 %: the bound is twice the norm 0.1");
    const AdaptiveReduction partial = Z.reduce_adaptive(0.05);
    check(test::close(partial.Z.generators(), Tensor({{0.1, 1.0, 0.0}, {0.1, 0.0, 1.0}}), 1e-12),
          b + ": 5 %: the oblique generator stays, then the box of the axis ones");
    check(partial.idx == std::vector<int64_t>({0, 1}) && partial.error == 0.0,
          b + ": 5 %: only the axis generators are boxed");
    check(test::close(partial.Z.center(), Z.center(), 0), b + ": the center stays");
}

/// An invalid percentage is refused, and with the checks off the reduction is the same.
void checks_and_their_switch() {
    const Zonotope Z(Tensor({0.0, 0.0}), Tensor({{1.0, 0.1}, {0.0, 0.1}}));
    check(test::raises([&] { Z.reduce_adaptive(-0.1); }, Error::wrong_value),
          "reduce_adaptive: < 0");
    const Tensor on = Z.reduce_adaptive(0.1).Z.generators();
    ChecksScope off(false);
    check(test::close(Z.reduce_adaptive(0.1).Z.generators(), on, 0),
          "reduce_adaptive: the same with the checks off");
}

/// A zonotope without generators stays; the result always contains the original, and applying
/// the indices of an adaptive reduction as "idx" gives the same set.
void reduction_encloses_and_idx_repeats_it(const std::string &b) {
    const Zonotope point(Tensor({1.0, 2.0}), Tensor::zeros({2, 0}));
    check(point.reduce_adaptive(0.1).Z.generators().shape()[1] == 0, b + ": no generators");
    cora::Rng rng(11);
    for (int run = 0; run < 20; ++run) {
        const Zonotope Z = Zonotope::generate_random(rng, 3, std::nullopt, 12);
        const AdaptiveReduction r = Z.reduce_adaptive(0.02 * (run + 1));
        check(test::close(Z.reduce_idx(r.idx).generators(), r.Z.generators(), 1e-12),
              b + ": idx repeats the adaptive reduction");
        for (int k = 0; k < 10; ++k) {
            const std::vector<double> d = test::random_direction(rng, 3);
            check(test::support(r.Z, d) >= test::support(Z, d) - 1e-9, b + ": it contains Z");
        }
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        a_hand_case_in_two_dimensions(b);
        reduction_encloses_and_idx_repeats_it(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope reduce adaptive");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
