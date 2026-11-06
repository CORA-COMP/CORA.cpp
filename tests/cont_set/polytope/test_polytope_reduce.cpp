// test_polytope_reduce - Polytope::reduce: an enclosure by random halfspaces
//
// CORA: contSet/@polytope/reduce.m
// PROPERTY: contSet/@polytope/reduce.m
//
// Property: the result contains the set and has at most 2 order n rows
//
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polytope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The number of rows and the enclosure of a polytope.
void reductions(const std::string &b) {
    Rng rng(4);
    const Polytope P = test::random_polytope(rng, 3, 8);
    for (int order = 1; order <= 2; ++order) {
        const Polytope R = P.reduce(order, rng);
        check(R.A().shape()[0] <= 6 * order && R.dim() == 3, b + ": at most 2 order n rows");
        check(R.contains(test::sample_points(P, rng, 20), "exact", 1e-8),
              b + ": it contains the set");
    }
    // the empty set stays empty
    check(Polytope::empty(2).reduce(1, rng).representsa("emptySet"), b + ": the empty set");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(reductions);
    Rng rng(4);
    check(test::raises([&] { test::square_h().reduce(0, rng); }, Error::wrong_value),
          "reduce: order >= 1");
    check(test::raises([&] { test::square_h().reduce(1, rng, "girard"); }, Error::wrong_value),
          "reduce: the method");
    ChecksScope off(false);
    check(test::square_h().reduce(1, rng).dim() == 2, "reduce: the same with the checks off");
    return test::finish("polytope reduce");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
