// test_polytope_less_equal - Polytope::less_equal: whether a polytope is contained in a set
//
// CORA: contSet/@polytope/le.m
// PROPERTY: contSet/@polytope/le.m
//
// Property: P <= S exactly if every vertex of P is in S
//
// CORA authors: Viktor Kotsev
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/fullspace/fullspace.h"
#include "polytope_testing.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Containment in polytopes, boxes, zonotopes and the special sets.
void containments(const std::string &b) {
    const Polytope big(Interval(column({-1.0, -1.0}), column({2.0, 2.0})));
    check(test::square_h().less_equal(big) && !big.less_equal(test::square_h()),
          b + ": a polytope");
    check(test::square_v().less_equal(Interval(column({-1.0, -1.0}), column({2.0, 2.0}))) &&
              !test::square_v().less_equal(Interval(column({0.5, 0.0}), column({2.0, 2.0}))),
          b + ": a box");
    check(test::square_h().less_equal(Zonotope(column({0.5, 0.5}), Tensor::eye(2))) &&
              !test::square_h().less_equal(Zonotope(column({0.5, 0.5}), Tensor::eye(2) * 0.2)),
          b + ": a zonotope");
    // the whole space holds all, the empty set only itself; an unbounded set is in no bounded set
    check(test::square_h().less_equal(Fullspace(2)) &&
              Polytope::empty(2).less_equal(test::square_h()) &&
              !test::square_h().less_equal(EmptySet(2)) &&
              Polytope::empty(2).less_equal(EmptySet(2)),
          b + ": R^n and the empty set");
    check(!Polytope(Tensor({{1.0, 0.0}}), column({1.0}))
               .less_equal(Interval(column({-9.0, -9.0}), column({9.0, 9.0}))),
          b + ": an unbounded set");
}

/// P <= S exactly if the vertices of P are in S.
void properties(const std::string &b) {
    Rng rng(4);
    bool same = true;
    for (int k = 0; k < 6; ++k) {
        const Polytope P1(test::sphere_points(rng, 2, 5)), P2 = test::random_polytope(rng, 2, 3);
        bool vertices = true;
        for (int64_t c = 0; c < P1.V().shape()[1]; ++c)
            vertices = vertices && P2.contains(P1.V().slice(-1, c, c + 1), "exact", 1e-9);
        same = same && P1.less_equal(P2, 1e-9) == vertices;
    }
    check(same, b + ": P <= S exactly if the vertices of P are in S");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        containments(b);
        properties(b);
    });
    check(test::raises([] { test::square_h().less_equal(Polytope::inf(3)); },
                       Error::dimension_mismatch),
          "less_equal: dimensions");
    check(test::raises([] { test::square_h().less_equal(test::square_h(), -1.0); },
                       Error::wrong_value),
          "less_equal: tol >= 0");
    ChecksScope off(false);
    check(test::square_h().less_equal(test::square_h()),
          "less_equal: the same with the checks off");
    return test::finish("polytope less_equal");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
