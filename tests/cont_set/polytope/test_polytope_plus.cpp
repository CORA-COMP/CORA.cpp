// test_polytope_plus - Polytope::plus: the Minkowski sum with a polytope, a box or a point
//
// CORA: contSet/@polytope/plus.m
// CORA: contSet/@polytope/private/priv_plus_minus_vector.m
// PROPERTY: contSet/@polytope/plus.m
//
// Property: the sum of sampled points of both sets lies in the sum, and the support functions add
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

/// Sums of both forms with a point, a box and a polytope.
void sums(const std::string &b) {
    const Polytope sq = test::square_h();
    // a point shifts the offsets, in both forms
    check(close(sq.plus(column({1.0, 2.0})).b(), {2.0, 3.0, -1.0, -2.0}),
          b + ": halfspaces + point");
    check(close(test::square_v().plus(column({1.0, 2.0})).V(),
                {1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 3.0, 3.0}),
          b + ": vertices + point");
    // two squares: [0, 2]^2 by their halfspaces
    const Polytope two = sq.plus(sq);
    check(two.contains(column({2.0, 2.0})) && two.contains(column({0.0, 0.0})) &&
              !two.contains(column({2.1, 1.0})) && close(test::support(two, {1.0, 1.0}), 4.0, 1e-8),
          b + ": H + H: the square plus the square");
    // vertices: all pairs, the second varying slowest
    const Polytope V = test::square_v().plus(Polytope(Tensor({{10.0, 20.0}, {0.0, 0.0}})));
    check(V.V().shape() == Shape{2, 8} && close(V.V().slice(-1, 0, 1), {10.0, 0.0}),
          b + ": V + V: all pairs, the second set varying slowest");
    // a box and a mixed pair
    check(sq.plus(Interval(column({1.0, 1.0}), column({2.0, 2.0}))).contains(column({3.0, 3.0})),
          b + ": + box");
    check(test::square_v().plus(sq).contains(column({2.0, 2.0})), b + ": V + H");
}

/// The special sets: R^n absorbs, the empty set annihilates, the origin is neutral.
void special(const std::string &b) {
    const Polytope sq = test::square_h();
    check(sq.plus(Polytope::inf(2)).representsa("fullspace") &&
              Polytope::inf(2).plus(sq).representsa("fullspace"),
          b + ": R^n");
    check(sq.plus(Polytope::empty(2)).representsa("emptySet"), b + ": the empty set");
    check(sq.plus(Polytope::origin(2)).isequal(sq), b + ": the origin");
}

/// Sampled sums lie in the sum, and the support functions add.
void properties(const std::string &b) {
    Rng rng(4);
    bool inside = true, added = true;
    for (int n = 2; n <= 3; ++n) {
        const Polytope P1 = test::random_polytope(rng, n, n + 1),
                       P2 = test::random_polytope(rng, n, n + 1);
        const Polytope S = P1.plus(P2);
        const Tensor x = test::sample_points(P1, rng, 8), y = test::sample_points(P2, rng, 8);
        inside = inside && S.contains(x + y, "exact", 1e-6);
        const std::vector<double> d = test::random_direction(rng, n);
        added =
            added && close(test::support(S, d), test::support(P1, d) + test::support(P2, d), 1e-6);
    }
    check(inside && added, b + ": samples are in the sum, support functions add");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        sums(b);
        special(b);
        properties(b);
    });
    check(test::raises([] { test::square_h().plus(Polytope::inf(3)); }, Error::dimension_mismatch),
          "plus: dimensions");
    check(test::raises([] { test::square_h().plus(Tensor::zeros({3, 1})); },
                       Error::dimension_mismatch),
          "plus: the point has the dimension");
    ChecksScope off(false);
    check(close(test::square_h().plus(column({1.0, 1.0})).b(), {2.0, 2.0, -1.0, -1.0}),
          "plus: the same with the checks off");
    return test::finish("polytope plus");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
