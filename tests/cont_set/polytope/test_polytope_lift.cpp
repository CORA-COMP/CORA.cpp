// test_polytope_lift - Polytope::lift: a polytope in a larger space
//
// CORA: contSet/@polytope/lift_.m
// PROPERTY: contSet/@polytope/lift_.m
//
// Property: a point is in the lifted set exactly if its chosen coordinates are in the set
//
// CORA authors: Viktor Kotsev, Mark Wetzlinger
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

/// The rows keep their columns at the chosen positions; the others are free.
void lifts(const std::string &b) {
    const Polytope P = test::square_h().lift(3, {2, 0});
    check(close(P.A(), {0, 0, 1, 1, 0, 0, 0, 0, -1, -1, 0, 0}),
          b + ": the columns move to 2 and 0");
    check(P.dim() == 3 && P.contains(column({0.5, 100.0, 0.5})) &&
              !P.contains(column({2.0, 0.0, 0.5})),
          b + ": the free coordinate is free");
    // N = n permutes a vertex set: coordinate j goes to dims[j]
    check(close(test::square_v().lift(2, {1, 0}).V(), {0, 0, 1, 1, 0, 1, 1, 0}),
          b + ": vertices, N = n");
    // vertices of a larger space are unbounded: the halfspaces are returned
    check(test::square_v().lift(3, {0, 1}).is_h_rep(), b + ": vertices lifted to more dimensions");
}

/// Points of the lifted set are exactly those whose chosen coordinates are in P.
void properties(const std::string &b) {
    Rng rng(4);
    bool same = true;
    const Polytope P = test::random_polytope(rng, 2, 4);
    const Polytope L = P.lift(4, {3, 1});
    for (int k = 0; k < 40; ++k) {
        std::vector<double> x = test::random_direction(rng, 4);
        same = same && L.contains(column(x)) == P.contains(column({x[3], x[1]}));
    }
    check(same, b + ": x in lift(P) exactly if (x3, x1) in P");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        lifts(b);
        properties(b);
    });
    check(test::raises([] { test::square_h().lift(1, {0, 1}); }, Error::wrong_value),
          "lift: the larger space is not smaller");
    check(test::raises([] { test::square_h().lift(3, {0}); }, Error::wrong_value),
          "lift: one position per dimension");
    check(test::raises([] { test::square_h().lift(3, {1, 1}); }, Error::wrong_value),
          "lift: a repeated position");
    ChecksScope off(false);
    check(test::square_h().lift(3, {0, 1}).dim() == 3, "lift: the same with the checks off");
    return test::finish("polytope lift");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
