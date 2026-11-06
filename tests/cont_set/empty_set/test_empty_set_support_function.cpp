// test_empty_set_support_function - EmptySet support_function: -Inf, the supremum over no point
//
// CORA: contSet/@emptySet/supportFunc_.m
// PROPERTY: contSet/@emptySet/supportFunc_.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"

#include <limits>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

const double kInf = std::numeric_limits<double>::infinity();

void the_support_function_is_minus_infinity(const std::string &b) {
    const EmptySet E(2);
    check(test::support(E, {1.0, 2.0}) == -kInf, b + ": -Inf");
    const Tensor batch = Tensor::from_data({1, 0, 0, 1, 1, 1}, {3, 2, 1});
    const Tensor rho = E.support_function(batch);
    check(rho.shape() == std::vector<int64_t>({3, 1, 1}) && rho.min().item() == -kInf &&
              rho.max().item() == -kInf,
          b + ": a batch of three directions");
    check(EmptySet(0).support_function(Tensor::zeros({0, 1})).data()[0] == -kInf,
          b + ": dimension 0");
}

/// The empty set lies in every set, so its support function bounds nothing above that of any set
/// (random zonotopes and directions of a fixed seed), and it is -Inf in each direction.
void the_empty_set_lies_below_every_set(const std::string &b) {
    Rng rng(11);
    for (int trial = 0; trial < 20; ++trial) {
        const int n = 1 + trial % 4;
        const Zonotope Z = Zonotope::generate_random(rng, n, std::nullopt, 3);
        const std::vector<double> d = test::random_direction(rng, n);
        check(test::support(EmptySet(n), d) <= test::support(Z, d) &&
                  test::support(EmptySet(n), d) == -kInf,
              b + ": at most the support function of a zonotope, dimension " + std::to_string(n));
    }
}

/// A direction of the wrong dimension is refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const EmptySet E(2);
    check(test::raises([&] { E.support_function(test::column({1.0, 2.0, 3.0})); },
                       Error::dimension_mismatch),
          "support_function: a direction of 3 dimensions");
    ChecksScope off(false);
    check(test::support(E, {1.0, 0.0}) == -kInf, "support_function: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(the_support_function_is_minus_infinity);
    test::for_each_backend(the_empty_set_lies_below_every_set);
    checks_and_their_switch();
    return test::finish("empty_set support_function");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
