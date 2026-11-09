// test_zonotope_linear_combination - zonotope linear_combination: the enclosure of the segments
//
// between two zonotopes
//
// CORA: contSet/@contSet/linComb.m
// CORA authors: Tobias Ladner
// Authors:       Tobias Ladner, Claude Sonnet 5.5, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   09-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <algorithm>

using namespace cora;
using test::check;
using test::close;
using test::column;
using test::random_direction;
using test::support;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// `linear_combination(Z, M Z + t)` has `2m + 1` generators and encloses both sets.
void linear_combination_encloses_both(const std::string &b) {
    cora::Rng rng(5);
    const int n = 3, m = 4;
    const Zonotope Z = Zonotope::generate_random(rng, n, std::nullopt, m);
    std::vector<double> M(n * n), t(n);
    rng.normal(M.data(), M.size(), 0.5);
    rng.normal(t.data(), t.size(), 1.0);
    const Zonotope mapped = Z.linear_map(Tensor::from_data(M, {n, n}));
    const Zonotope Z2(mapped.center() + column(t), mapped.generators());
    const Zonotope hull = Z.linear_combination(Z2);
    check(hull.generators().shape() == std::vector<int64_t>({n, 2 * m + 1}),
          b + ": 2m + 1 generators");
    for (int k = 0; k < 20; ++k) {
        const std::vector<double> d = random_direction(rng, n);
        check(support(hull, d) >= std::max(support(Z, d), support(Z2, d)) - 1e-10,
              b + ": linear_combination leaves one of its sets");
    }
    check(close(Z.linear_combination(Z).interval().inf(), Z.interval().inf(), 1e-12),
          b + ": linear_combination(Z, Z) is Z");
    // two points: the segment between them, the shift its one generator
    const Zonotope p(column({0.0, 2.0}), Tensor::zeros({2, 0})),
        q(column({4.0, 2.0}), Tensor::zeros({2, 0}));
    const Zonotope seg = p.linear_combination(q);
    check(close(seg.center(), {2, 2}) && close(seg.generators(), {-2, 0}),
          b + ": the segment between two points");
}

/// Zonotopes of different shapes are refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const Zonotope Z = Zonotope(Tensor({1.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}})),
                   Z3 = Zonotope(Tensor({1.0, 0.0, 0.0}),
                                 Tensor({{0.5, 0.0}, {0.0, 0.5}, {0.0, 0.0}}));
    check(test::raises([&] { Z.linear_combination(Z3); }, Error::dimension_mismatch),
          "linear_combination: dimensions 2 and 3");
    const Tensor on = Z.linear_combination(Z).generators();
    ChecksScope off(false);
    check(test::close(Z.linear_combination(Z).generators(), on, 0),
          "linear_combination: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { linear_combination_encloses_both(b); });
    checks_and_their_switch();
    return test::finish("zonotope linear_combination");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
