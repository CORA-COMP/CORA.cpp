// test_zonotope_linear_combination - zonotope linear_combination: the enclosure of the segments
// between two zonotopes
//
// CORA: contSet/@contSet/linComb.m

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

int main() {
    test::for_each_backend([](const std::string &b) { linear_combination_encloses_both(b); });
    checks_and_their_switch();
    return test::finish("zonotope linear_combination");
}
