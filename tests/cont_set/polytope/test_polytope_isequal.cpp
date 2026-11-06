// test_polytope_isequal - Polytope::isequal: whether a polytope equals a polytope, a point or a set
//
// CORA: contSet/@polytope/isequal.m
// PROPERTY: contSet/@polytope/isequal.m
//
// Property: a polytope equals its vertex form and its rescaled rows, and no shifted copy
//
// CORA authors: Viktor Kotsev, Mark Wetzlinger, Adrian Kulmburg
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

/// Polytopes in the two forms, with redundant or scaled rows, and different sets.
void polytopes(const std::string &b) {
    const Polytope sq = test::square_h();
    check(sq.isequal(sq) && sq.isequal(test::square_v()) && test::square_v().isequal(sq),
          b + ": the square");
    // the same set with scaled rows and a redundant row
    const Polytope scaled(Tensor({{2.0, 0.0}, {0.0, 3.0}, {-1.0, 0.0}, {0.0, -4.0}, {1.0, 1.0}}),
                          column({2.0, 3.0, 0.0, 0.0, 5.0}));
    check(sq.isequal(scaled) && scaled.isequal(sq), b + ": scaled rows, a redundant row");
    check(!sq.isequal(test::triangle_h()) && !sq.isequal(sq.plus(column({1.0, 0.0}))),
          b + ": different sets");
    // flat sets by containment, R^n and the empty set
    const Polytope segment(Tensor({{0.0, 1.0}, {0.0, 1.0}}));
    check(segment.isequal(segment) && !segment.isequal(Polytope(Tensor({{0.0, 2.0}, {0.0, 2.0}}))),
          b + ": segments");
    check(Polytope::inf(2).isequal(Polytope::inf(2)) && !Polytope::inf(2).isequal(sq) &&
              !sq.isequal(Polytope::inf(2)),
          b + ": R^n");
    check(Polytope::empty(2).isequal(Polytope::empty(2)) && !Polytope::empty(2).isequal(sq),
          b + ": the empty set");
    check(!sq.isequal(Polytope::inf(3)), b + ": different dimensions");
    // one dimension
    check(Polytope(Tensor({{-1.0}, {1.0}}), column({0.0, 1.0}))
              .isequal(Polytope(Tensor({{0.0, 1.0}}))),
          b + ": 1D");
}

/// Points and other sets.
void others(const std::string &b) {
    check(Polytope(column({1.0, 2.0})).isequal(column({1.0, 2.0})) &&
              !Polytope(column({1.0, 2.0})).isequal(column({1.0, 3.0})) &&
              !test::square_h().isequal(column({0.5, 0.5})),
          b + ": a point");
    check(test::square_h().isequal(Interval(column({0.0, 0.0}), column({1.0, 1.0}))) &&
              !test::square_h().isequal(Interval(column({0.0, 0.0}), column({1.0, 2.0}))),
          b + ": a box");
    check(test::square_h().isequal(Zonotope(column({0.5, 0.5}), Tensor::eye(2) * 0.5)),
          b + ": a zonotope");
    check(Polytope::inf(2).isequal(Fullspace(2)) && Polytope::empty(2).isequal(EmptySet(2)),
          b + ": the special sets");
}

/// A polytope equals its vertex form, and not a shifted copy.
void properties(const std::string &b) {
    Rng rng(4);
    bool same = true, different = true;
    for (int n = 2; n <= 3; ++n) {
        const Polytope P = test::random_polytope(rng, n, n + 2);
        same = same && P.isequal(test::as_vertices(P)) && P.isequal(P.normalize_constraints("A"));
        different =
            different && !P.isequal(P.plus(column(std::vector<double>(std::size_t(n), 0.5))));
    }
    check(same && different,
          b + ": equal to its vertex form and rescaled rows, not to a shifted copy");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        polytopes(b);
        others(b);
        properties(b);
    });
    check(
        test::raises([] { test::square_h().isequal(test::square_h(), -1.0); }, Error::wrong_value),
        "isequal: tol >= 0");
    check(test::raises(
              [] {
                  Polytope(Tensor::ones({3, 2, 2}), Tensor::ones({3, 2, 1}))
                      .isequal(test::square_h());
              },
              Error::not_supported),
          "isequal: a batch");
    ChecksScope off(false);
    check(test::square_h().isequal(test::square_h()), "isequal: the same with the checks off");
    return test::finish("polytope isequal");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
