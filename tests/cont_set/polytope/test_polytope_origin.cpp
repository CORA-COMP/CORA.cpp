// test_polytope_origin - Polytope::origin: the polytope that is the origin
//
// CORA: contSet/@polytope/origin.m
// PROPERTY: contSet/@polytope/origin.m
//
// Property: the origin is the only point, in both forms
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

/// The data of the origin and the points it holds.
void origin_sets(const std::string &b) {
    const Polytope P = Polytope::origin(3);
    check(P.dim() == 3 && P.is_h_rep() && P.is_v_rep(), b + ": both forms, dimension 3");
    check(close(P.V(), {0, 0, 0}, 0) && P.A().shape() == Shape{0, 3},
          b + ": one vertex 0, no rows");
    check(close(P.Ae(), Tensor::eye(3), 0) && close(P.be(), {0, 0, 0}, 0),
          b + ": the equalities x = 0");
    // exactly the origin
    Rng rng(4);
    bool only = P.contains(column({0.0, 0.0, 0.0}));
    for (int k = 0; k < 20; ++k) {
        std::vector<double> x = test::random_direction(rng, 3);
        only = only && !P.contains(column(x));
    }
    check(only, b + ": it holds the origin and no other point");
    check(P.interval().representsa("origin"), b + ": its box is the origin");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(origin_sets);
    check(test::raises([] { Polytope::origin(0); }, Error::wrong_value), "origin: n >= 1");
    // with the checks off
    ChecksScope off(false);
    check(Polytope::origin(2).dim() == 2, "origin: the same with the checks off");
    return test::finish("polytope origin");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
