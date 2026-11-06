// test_zonotope_plus - zonotope plus: the Minkowski sum
//
// CORA: contSet/@zonotope/plus.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner, Claude Sonnet 5.5, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::random_direction;
using test::support;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void plus_is_the_minkowski_sum(const std::string &b) {
    cora::Rng rng(4);
    const int n = 3;
    const Zonotope A = Zonotope::generate_random(rng, n, std::nullopt, 3),
                   B = Zonotope::generate_random(rng, n, std::nullopt, 2);
    const Zonotope S = A.plus(B);
    check(S.generators().shape() == std::vector<int64_t>({n, 5}), b + ": generators are joined");
    check(close(S.center(), A.center() + B.center()), b + ": centres add");
    for (int k = 0; k < 5; ++k) {
        const std::vector<double> d = random_direction(rng, n);
        check(close(support(S, d), support(A, d) + support(B, d), 1e-10), b + ": supports add");
    }
}

/// Sets of different dimensions are refused; with the checks off the sum is the same.
void checks_and_their_switch() {
    const Zonotope S = Zonotope(Tensor({1.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}})),
                   S3 = Zonotope(Tensor({1.0, 0.0, 0.0}),
                                 Tensor({{0.5, 0.0}, {0.0, 0.5}, {0.0, 0.0}}));
    check(test::raises([&] { S.plus(S3); }, Error::dimension_mismatch), "plus: dimensions 2 and 3");
    const Tensor on = S.plus(S).center();
    ChecksScope off(false);
    check(test::close(S.plus(S).center(), on, 0), "plus: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { plus_is_the_minkowski_sum(b); });
    checks_and_their_switch();
    return test::finish("zonotope plus");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
