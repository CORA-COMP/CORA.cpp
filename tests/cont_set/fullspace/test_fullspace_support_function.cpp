// test_fullspace_support_function - fullspace support_function: Inf along every nonzero direction
//
// CORA: contSet/@fullspace/supportFunc_.m
// PROPERTY: contSet/@fullspace/supportFunc_.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"

#include <limits>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

const double kInf = std::numeric_limits<double>::infinity();

void the_support_function_is_infinite_unless_the_direction_is_zero(const std::string &b) {
    const Fullspace F(3);
    check(test::support(F, {1.0, 0.0, 0.0}) == kInf, b + ": along an axis");
    check(test::support(F, {0.0, -2.0, 1e-300}) == kInf, b + ": a tiny entry still counts");
    check(test::support(F, {0.0, 0.0, 0.0}) == 0.0, b + ": the zero direction gives 0 (CORALean)");
    const Tensor batch = Tensor::from_data({1, 0, 0, 0, 0, 0, 0, 1, 1}, {3, 3, 1});
    const Tensor rho = F.support_function(batch);
    check(rho.shape() == std::vector<int64_t>({3, 1, 1}) && rho.data()[0] == kInf &&
              rho.data()[1] == 0.0 && rho.data()[2] == kInf,
          b + ": a batch of three directions");
}

/// The whole space contains every set, so its support function is at least that of any zonotope
/// (random zonotopes and directions of a fixed seed).
void the_whole_space_lies_above_every_set(const std::string &b) {
    Rng rng(12);
    for (int trial = 0; trial < 20; ++trial) {
        const int n = 1 + trial % 4;
        const Zonotope Z = Zonotope::generate_random(rng, n, std::nullopt, 3);
        const std::vector<double> d = test::random_direction(rng, n);
        check(test::support(Fullspace(n), d) >= test::support(Z, d),
              b + ": at least the support function of a zonotope, dimension " + std::to_string(n));
    }
}

/// A direction of the wrong dimension is refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const Fullspace F(2);
    check(test::raises([&] { F.support_function(test::column({1.0, 2.0, 3.0})); },
                       Error::dimension_mismatch),
          "support_function: a direction of 3 dimensions");
    ChecksScope off(false);
    check(test::support(F, {1.0, 0.0}) == kInf, "support_function: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(the_support_function_is_infinite_unless_the_direction_is_zero);
    test::for_each_backend(the_whole_space_lies_above_every_set);
    checks_and_their_switch();
    return test::finish("fullspace support_function");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
