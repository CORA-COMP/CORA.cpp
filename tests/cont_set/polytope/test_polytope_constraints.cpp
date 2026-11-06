// test_polytope_constraints - Polytope::constraints: the halfspaces of either representation
//
// CORA: contSet/@polytope/constraints.m
// PROPERTY: contSet/@polytope/constraints.m
// CORA authors: Viktor Kotsev
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

// Property: the halfspaces hold exactly the vertices and nothing beyond them (hand-computed
//           rows, and random clouds: the points are inside, the facets are tight at their
//           vertices)

#include "polytope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A halfspace polytope is its own constraints.
void from_halfspaces(const std::string &b) {
    const Polytope P(Tensor({{1.0, 0.0}, {0.0, 1.0}}), column({1.0, 2.0}), Tensor({{1.0, 1.0}}),
                     column({0.5}));
    const Polytope::Constraints C = P.constraints();
    check(close(C.A, P.A(), 0) && close(C.b, P.b(), 0) && close(C.Ae, P.Ae(), 0) &&
              close(C.be, P.be(), 0),
          b + ": halfspaces are returned as held");
}

/// The unit square: four facets, each once with a unit normal.
void from_square(const std::string &b) {
    const Polytope::Constraints C =
        Polytope(Tensor({{0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}})).constraints();
    check(C.A.shape() == Shape{4, 2} && C.Ae.shape() == Shape{0, 2},
          b + ": the square has 4 facets");
    check(test::inside_halfspaces(Tensor({{0.5, 0.0, 1.0, 0.2}, {0.5, 0.3, 1.0, 1.0}}), C),
          b + ": the points of the square satisfy the rows");
    check(!test::inside_halfspaces(Tensor({{1.5}, {0.5}}), C),
          b + ": a point outside violates one");
    check(close(C.A.square().sum(-1, true), {1, 1, 1, 1}), b + ": unit normals");
}

/// The degenerate sets: a point, a segment, a flat set, the line.
void degenerate(const std::string &b) {
    const Polytope::Constraints point = Polytope(column({1.0, 2.0})).constraints();
    check(point.A.shape() == Shape{0, 2} && close(point.Ae, {1, 0, 0, 1}) &&
              close(point.be, {1, 2}),
          b + ": a point: the equalities I x = v");
    const Polytope::Constraints bounds = Polytope(Tensor({{3.0, 1.0, 2.0}})).constraints();
    check(close(bounds.A, {1, -1}) && close(bounds.b, {3, -1}), b + ": 1D: the bounds");
    const Polytope::Constraints segment = Polytope(Tensor({{0.0, 2.0}, {0.0, 2.0}})).constraints();
    check(segment.A.shape()[0] == 4 && test::inside_halfspaces(Tensor({{1.0}, {1.0}}), segment) &&
              !test::inside_halfspaces(Tensor({{1.0}, {0.0}}), segment) &&
              !test::inside_halfspaces(Tensor({{3.0}, {3.0}}), segment),
          b + ": a diagonal segment: the line as two rows and the bounds");
    const Polytope::Constraints flat =
        Polytope(Tensor({{0.0, 1.0, 0.0, 1.0}, {0.0, 0.0, 1.0, 1.0}, {1.0, 1.0, 1.0, 1.0}}))
            .constraints();
    check(flat.Ae.shape()[0] == 1 && test::inside_halfspaces(Tensor({{0.5}, {0.5}, {1.0}}), flat) &&
              !test::inside_halfspaces(Tensor({{0.5}, {0.5}, {0.0}}), flat),
          b + ": a square in space: an equality for its plane");
    const Polytope empty = Polytope::empty(2);
    // the empty set: 0 <= -1
    check(close(empty.constraints().b, {-1}), b + ": the empty set: 0 <= -1");
}

/// Random clouds: every vertex satisfies the rows, and every row is tight at some vertex.
void clouds(const std::string &b) {
    Rng rng(7);
    for (const int n : {2, 3, 4}) {
        const Tensor points = test::sphere_points(rng, n, n + 4);
        const Polytope::Constraints C = Polytope(points).constraints();
        check(test::inside_halfspaces(points, C),
              b + ": " + std::to_string(n) + "D: vertices inside");
        const Tensor slack = C.b - C.A.matmul(points);
        check(slack.min(-1).abs().max().item() < 1e-8,
              b + ": " + std::to_string(n) + "D: facets tight");
        check(!test::inside_halfspaces(points * 1.01, C),
              b + ": " + std::to_string(n) + "D: scaled out");
    }
}

/// A batch of vertex polytopes is not supported; the other input errors; checks off.
void refusals() {
    check(test::raises([&] { Polytope(Tensor::ones({2, 2, 3})).constraints(); },
                       Error::not_supported),
          "constraints: a batch of vertex polytopes");
    const Polytope P(Tensor({{0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}}));
    ChecksScope off(false);
    check(P.constraints().A.shape() == Shape{4, 2}, "constraints: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        from_halfspaces(b);
        from_square(b);
        degenerate(b);
        clouds(b);
    });
    refusals();
    return test::finish("polytope constraints");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
