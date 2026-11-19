// test_polytope_intersect - Polytope::intersect: the intersection with a polytope, a box or a point
//
// CORA: contSet/@polytope/and_.m
// PROPERTY: contSet/@polytope/and_.m
//
// Property: a point is in the intersection iff it is in both sets
//
// CORA authors: Viktor Kotsev, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "polytope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Polytopes: the rows stack; R^n is neutral; boxes; points.
void intersections(const std::string &b) {
    const Polytope P(Tensor({{1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}}),
                     vec({2.0, 2.0, 0.0, 0.0}));
    const Polytope Q(Tensor({{-1.0, 0.0}, {0.0, -1.0}, {1.0, 0.0}, {0.0, 1.0}}),
                     vec({-1.0, -1.0, 3.0, 3.0}));
    const Polytope R = P.intersect(Q);
    check(R.A().shape() == Shape{8, 2}, b + ": the rows stack");
    check(R.contains(vec({1.5, 1.5})) && !R.contains(vec({0.5, 1.5})) &&
              !R.contains(vec({2.5, 1.5})),
          b + ": the points");
    check(close(P.intersect(Polytope::inf(2)).A(), P.A(), 0), b + ": R^n on the right");
    check(close(Polytope::inf(2).intersect(P).A(), P.A(), 0), b + ": R^n on the left");
    const Polytope S = P.intersect(Interval(vec({1.0, 1.0}), vec({5.0, 5.0})));
    check(S.contains(vec({1.5, 1.5})) && !S.contains(vec({0.5, 1.5})), b + ": a box");
    check(P.intersect(Interval::empty(2)).interval().is_empty() ||
              !P.intersect(Interval::empty(2)).contains(vec({1.0, 1.0})),
          b + ": the empty box");
    const Polytope pt = P.intersect(vec({1.0, 1.0}));
    check(pt.is_v_rep() && close(pt.V(), {1, 1}, 0), b + ": a point inside: the point");
    // a point inside, outside
    check(P.intersect(vec({3.0, 1.0})).V().shape() == Shape{0, 2}, b + ": a point outside: empty");
    check(P.intersect(vec({2.0 + 1e-6, 1.0}), 1e-5).V().size(-2) != 0, b + ": a tolerance");
    // vertex polytopes are intersected as their halfspaces
    const Polytope V(Tensor({{0.0, 0.0}, {3.0, 0.0}, {3.0, 3.0}, {0.0, 3.0}}));
    check(V.intersect(P).contains(vec({1.0, 1.0})) && !V.intersect(P).contains(vec({2.5, 1.0})),
          b + ": vertices");
}

/// Batches stack member by member.
void batches(const std::string &b) {
    const Polytope A(Tensor::ones({3, 2, 2}), Tensor::ones({3, 2}));
    check(A.intersect(A).A().shape() == Shape{3, 4, 2}, b + ": a batch of three");
}

/// What is refused; checks off.
void refusals() {
    const Polytope S(Tensor::ones({1, 2}), Tensor::ones({1}));
    const Polytope A(Tensor::ones({3, 2, 2}), Tensor::ones({3, 2}));
    check(test::raises([&] { S.intersect(Polytope::inf(3)); }, Error::dimension_mismatch),
          "intersect: dimensions");
    check(test::raises([&] { S.intersect(A); }, Error::dimension_mismatch),
          "intersect: batch shapes");
    check(test::raises([&] { S.intersect(Tensor::zeros({3, 1})); }, Error::dimension_mismatch),
          "intersect: the point has the dimension");
    ChecksScope off(false);
    check(S.intersect(S).A().shape() == Shape{2, 2}, "intersect: the same with the checks off");
}

/// A probe is in the intersection iff it is in both sets.
void properties(const std::string &b) {
    Rng rng(4);
    for (const int n : {2, 3}) {
        const Polytope P1 = test::random_polytope(rng, n, 2 * n),
                       P2 = test::random_polytope(rng, n, 2 * n);
        const Polytope both = P1.intersect(P2);
        bool same = true;
        for (int k = 0; k < 60; ++k) {
            std::vector<double> x(static_cast<std::size_t>(n));
            rng.uniform(x.data(), x.size(), -3.5, 3.5);
            const Tensor p = vec(x);
            same = same && both.contains(p) == (P1.contains(p) && P2.contains(p));
        }
        check(same, b + ": exactly the common points, " + std::to_string(n) + "D");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        properties(b);
        intersections(b);
        batches(b);
    });
    refusals();
    return test::finish("polytope intersect");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
