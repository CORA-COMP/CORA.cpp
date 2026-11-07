// test_polytope_minkowski_difference - Polytope::minkowski_difference: with a set or a point
//
// CORA: contSet/@polytope/minkDiff.m
// CORA: contSet/@polytope/private/priv_minkDiff.m
// PROPERTY: contSet/@polytope/minkDiff.m
//
// Property: the difference plus S lies in P, and every point that does so is in the difference
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

/// Differences with a box, a point, a zonotope, in both types.
void differences(const std::string &b) {
    const Interval box(column({-0.25, -0.25}), column({0.25, 0.25}));
    const Set diff = test::square_h().minkowski_difference(box);
    check(diff->class_name() == "Polytope", b + ": the result is a polytope");
    const Polytope &D = diff.as<Polytope>();
    check(D.contains(column({0.25, 0.25})) && D.contains(column({0.75, 0.75})) &&
              !D.contains(column({0.2, 0.5})) && !D.contains(column({0.8, 0.5})),
          b + ": the square minus a box: [0.25, 0.75]^2");
    // a point shifts
    check(
        close(test::square_h().minkowski_difference(column({0.5, 0.5})).b(), {0.5, 0.5, 0.5, 0.5}),
        b + ": a point");
    // too large: empty
    const Interval large(column({-1.0, -1.0}), column({1.0, 1.0}));
    check(test::square_h().minkowski_difference(large).as<Polytope>().representsa("emptySet"),
          b + ": a larger set leaves nothing");
    // the other type, with the vertices of a polytope and the generators of a zonotope
    const Set by_vertices =
        test::square_h().minkowski_difference(box, std::string("exact:vertices"));
    check(by_vertices.as<Polytope>().contains(column({0.5, 0.5})) &&
              !by_vertices.as<Polytope>().contains(column({0.9, 0.5})),
          b + ": exact:vertices with a box");
    const Polytope small(Tensor({{-0.25, 0.25, 0.25, -0.25}, {-0.25, -0.25, 0.25, 0.25}}));
    const Set by_polytope =
        test::square_h().minkowski_difference(small, std::string("exact:vertices"));
    check(by_polytope.as<Polytope>().isequal(D),
          b + ": exact:vertices with a polytope is the same set");
    // R^n minus a set is R^n
    check(Polytope::inf(2).minkowski_difference(box).as<Polytope>().representsa("fullspace"),
          b + ": R^n");
}

/// The points of the difference plus the corners of the box lie in the set.
void properties(const std::string &b) {
    Rng rng(4);
    bool inside = true;
    for (int n = 2; n <= 3; ++n) {
        const Polytope P = test::random_polytope(rng, n, n + 2);
        const Interval small(Tensor::zeros({n, 1}) - 0.1, Tensor::zeros({n, 1}) + 0.1);
        const Polytope D = P.minkowski_difference(small).as<Polytope>();
        // a point of D plus any corner of the box is in P
        const Tensor x = test::sample_points(D, rng, 10);
        inside = inside && P.contains(x + 0.1, "exact", 1e-8) && P.contains(x - 0.1, "exact", 1e-8);
    }
    check(inside, b + ": D + S lies in P");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        differences(b);
        properties(b);
    });
    const Interval box(column({-0.25, -0.25}), column({0.25, 0.25}));
    check(test::raises([&] { test::square_h().minkowski_difference(box, std::string("inner")); },
                       Error::wrong_value),
          "minkowski_difference: the type");
    check(test::raises([&] { test::square_h().minkowski_difference(Polytope::inf(3)); },
                       Error::dimension_mismatch),
          "minkowski_difference: dimensions");
    ChecksScope off(false);
    check(test::square_h().minkowski_difference(column({0.5, 0.5})).dim() == 2,
          "minkowski_difference: the same with the checks off");
    return test::finish("polytope minkowski_difference");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
