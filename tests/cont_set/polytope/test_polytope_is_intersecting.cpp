// test_polytope_is_intersecting - Polytope::is_intersecting: whether the set meets another
//
// CORA: contSet/@polytope/isIntersecting_.m
// PROPERTY: contSet/@polytope/isIntersecting_.m
//
// Property: sets that share a sampled point meet; sets separated by a hyperplane do not
//
// CORA authors: Viktor Kotsev, Mark Wetzlinger, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/registry.h"
#include "polytope_testing.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Polytopes of the two forms, boxes, zonotopes, constrained zonotopes and bundles.
void sets(const std::string &b) {
    const Polytope sq = test::square_h();
    const Polytope near = sq.plus(column({0.5, 0.5})), far = sq.plus(column({3.0, 3.0}));
    check(sq.is_intersecting(near) && !sq.is_intersecting(far), b + ": H and H");
    check(test::square_v().is_intersecting(near) && !test::square_v().is_intersecting(far) &&
              sq.is_intersecting(test::as_vertices(near)) &&
              !sq.is_intersecting(test::as_vertices(far)),
          b + ": V and H");
    check(test::square_v().is_intersecting(test::as_vertices(near)) &&
              !test::square_v().is_intersecting(test::as_vertices(far)),
          b + ": V and V");
    // a box, a zonotope and a constrained zonotope, each meeting and missing
    check(sq.is_intersecting(Interval(column({0.5, 0.5}), column({2.0, 2.0}))) &&
              !sq.is_intersecting(Interval(column({2.0, 2.0}), column({3.0, 3.0}))),
          b + ": a box");
    check(sq.is_intersecting(Zonotope(column({1.0, 1.0}), Tensor::eye(2) * 0.6)) &&
              !sq.is_intersecting(Zonotope(column({3.0, 3.0}), Tensor::eye(2) * 0.6)),
          b + ": a zonotope");
    check(sq.is_intersecting(ConZonotope(column({1.0, 1.0}), Tensor::eye(2) * 0.6)) &&
              !sq.is_intersecting(ConZonotope(column({3.0, 3.0}), Tensor::eye(2) * 0.6)),
          b + ": a constrained zonotope");
    // a bundle meets if all its zonotopes meet at one point
    const ZonoBundle meets({Zonotope(column({1.0, 1.0}), Tensor::eye(2) * 0.6),
                            Zonotope(column({0.8, 1.0}), Tensor::eye(2))});
    const ZonoBundle misses({Zonotope(column({3.0, 3.0}), Tensor::eye(2) * 0.6),
                             Zonotope(column({3.2, 3.0}), Tensor::eye(2))});
    check(sq.is_intersecting(meets) && !sq.is_intersecting(misses), b + ": a bundle");
    // points: all must be inside
    check(sq.is_intersecting(Tensor({{0.5, 0.2}, {0.5, 0.9}})) &&
              !sq.is_intersecting(Tensor({{0.5, 1.2}, {0.5, 0.9}})),
          b + ": points");
}

/// Approximations, halfspaces, R^n, empty sets and one dimension.
void special(const std::string &b) {
    const Polytope sq = test::square_h();
    const Zonotope Z(column({1.0, 1.0}), Tensor::eye(2) * 0.6),
        Zfar(column({3.0, 3.0}), Tensor::eye(2) * 0.6);
    check(sq.is_intersecting(Z, "approx") && !sq.is_intersecting(Zfar, "approx"), b + ": approx");
    // a halfspace x <= 1 is met by a set that reaches into it
    const Polytope halfspace(Tensor({{1.0, 0.0}}), column({1.0}));
    check(halfspace.is_intersecting(Z) &&
              !halfspace.is_intersecting(Zonotope(column({3.0, 0.0}), Tensor::eye(2) * 0.5)) &&
              halfspace.is_intersecting(halfspace) && halfspace.is_intersecting(sq),
          b + ": a halfspace");
    check(Polytope(Tensor({{1.0, 0.0}}), column({1.0}))
                  .is_intersecting(Polytope(Tensor({{-1.0, 0.0}}), column({-2.0}))) == false,
          b + ": two halfspaces that face away");
    // R^n meets every set that is not empty; in 1D the intervals decide
    check(Polytope::inf(2).is_intersecting(sq) &&
              !Polytope::inf(2).is_intersecting(Polytope::empty(2)) &&
              !sq.is_intersecting(Polytope::empty(2)),
          b + ": R^n and the empty set");
    check(Polytope(Tensor({{-1.0}, {1.0}}), column({0.0, 1.0}))
                  .is_intersecting(Polytope(Tensor({{-1.0}, {1.0}}), column({-2.0, 3.0}))) == false,
          b + ": 1D");
}

/// The table holds the pairs of a polytope with a zonotope, a constrained zonotope and a bundle,
/// in both orders of the operands, and agrees with the members (exact by a linear program).
void table(const std::string &b) {
    const Polytope sq = test::square_h();
    const Zonotope Z(column({1.0, 1.0}), Tensor::eye(2) * 0.6),
        Zfar(column({3.0, 3.0}), Tensor::eye(2) * 0.6);
    check(is_intersecting(sq, Z) && is_intersecting(Z, sq) && !is_intersecting(sq, Zfar) &&
              !is_intersecting(Zfar, sq),
          b + ": a zonotope");
    check(is_intersecting(sq, ConZonotope(Z)) && !is_intersecting(ConZonotope(Zfar), sq),
          b + ": a constrained zonotope");
    check(is_intersecting(sq, ZonoBundle({Z, Z})) && !is_intersecting(sq, ZonoBundle({Zfar, Zfar})),
          b + ": a bundle");
}

/// Sets that share a sample meet; the translate by a large vector does not.
void properties(const std::string &b) {
    Rng rng(4);
    bool meet = true, apart = true;
    for (int n = 2; n <= 3; ++n) {
        const Polytope P = test::random_polytope(rng, n, n + 2);
        const Tensor x = test::sample_points(P, rng, 1);
        meet = meet && P.is_intersecting(Zonotope(x, Tensor::eye(n) * 0.1)) &&
               P.is_intersecting(Polytope(x));
        apart = apart && !P.is_intersecting(Zonotope(x + 100.0, Tensor::eye(n) * 0.1));
    }
    check(meet && apart, b + ": a shared sample meets; a far set does not");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        sets(b);
        special(b);
        table(b);
        properties(b);
    });
    // a set of another dimension and an unknown type are refused
    check(test::raises([] { test::square_h().is_intersecting(Polytope::inf(3)); },
                       Error::dimension_mismatch),
          "is_intersecting: dimensions");
    check(test::raises(
              [] {
                  test::square_h().is_intersecting(Zonotope(column({0.0, 0.0}), Tensor::eye(2)),
                                                   "inner");
              },
              Error::wrong_value),
          "is_intersecting: the type");
    ChecksScope off(false);
    check(test::square_h().is_intersecting(test::square_h()),
          "is_intersecting: the same with the checks off");
    return test::finish("polytope is_intersecting");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
