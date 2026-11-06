// test_polytope_project - Polytope::project: the projection onto some dimensions
//
// CORA: contSet/@polytope/project.m
// PROPERTY: contSet/@polytope/project.m
//
// Property: the projection of a sample is in the projection, which reaches as far as the set
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

/// Vertices are selected; halfspaces are eliminated.
void projections(const std::string &b) {
    // vertices: selection, repetition and reordering
    check(close(test::square_v().project({1, 0}).V(), {0, 0, 1, 1, 0, 1, 1, 0}),
          b + ": vertices, swapped");
    check(close(test::square_v().project({0}).V(), {0, 1}), b + ": vertices, one dimension");
    check(test::square_v().project({0, 0}).V().shape() == Shape{2, 4}, b + ": vertices, repeated");
    // halfspaces: the triangle onto x is [0, 2]; onto (y, x) is its mirror image
    const Polytope shadow = test::triangle_h().project({0});
    check(shadow.contains(column({0.0})) && shadow.contains(column({2.0})) &&
              !shadow.contains(column({-0.1})) && !shadow.contains(column({2.1})),
          b + ": halfspaces: the triangle onto x");
    const Polytope mirror = test::triangle_h().project({1, 0});
    check(mirror.contains(column({0.5, 1.5})) && !mirror.contains(column({1.5, 1.5})),
          b + ": halfspaces: the order of the dimensions");
    // a 3D wedge x + y + z <= 1, x, y, z >= 0 onto (x, y)
    const Polytope simplex(
        Tensor({{-1.0, 0.0, 0.0}, {0.0, -1.0, 0.0}, {0.0, 0.0, -1.0}, {1.0, 1.0, 1.0}}),
        column({0.0, 0.0, 0.0, 1.0}));
    const Polytope plane = simplex.project({0, 1});
    check(plane.contains(column({0.5, 0.5})) && !plane.contains(column({0.6, 0.6})) &&
              plane.contains(column({0.0, 1.0})),
          b + ": a 3D simplex onto the plane");
    // all dimensions in order: the set itself; the empty set stays empty
    check(close(test::square_h().project({0, 1}).A(), test::square_h().A(), 0),
          b + ": all dimensions");
    check(Polytope::empty(2).project({0}).representsa("emptySet"), b + ": the empty set");
}

/// Projected samples lie in the projection; its extreme values are those of the set.
void properties(const std::string &b) {
    Rng rng(4);
    bool inside = true, reach = true;
    for (int n = 3; n <= 4; ++n) {
        const Polytope P = test::random_polytope(rng, n, n + 1);
        const Polytope Q = P.project({0, 2});
        const Tensor x = test::sample_points(P, rng, 10);
        inside = inside && Q.contains(x.take({0, 2}, -2), "exact", 1e-6);
        // along (1, 1) the projection reaches as far as the set along (1, 0, 1, ...)
        std::vector<double> d(std::size_t(n), 0.0);
        d[0] = d[2] = 1.0;
        reach = reach && close(test::support(Q, {1.0, 1.0}), test::support(P, d), 1e-6);
    }
    check(inside && reach, b + ": samples project into the projection, which reaches as far");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        projections(b);
        properties(b);
    });
    check(test::raises([] { test::square_h().project({2}); }, Error::wrong_value),
          "project: a dimension outside");
    check(test::raises([] { test::square_h().project({0, 0}); }, Error::not_supported),
          "project: a repeated dimension of halfspaces");
    ChecksScope off(false);
    check(test::square_h().project({0}).dim() == 1, "project: the same with the checks off");
    return test::finish("polytope project");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
