// test_polytope_contains - Polytope::contains: points and sets
//
// CORA: contSet/@polytope/contains_.m
// PROPERTY: contSet/@polytope/contains_.m
//
// Property: the points of the set are contained and points outside are not; sets are contained
//           iff their support function stays below every row
//
// CORA authors: Niklas Kochdumper, Viktor Kotsev, Adrian Kulmburg, Mark Wetzlinger, Tobias Ladner
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/capsule/capsule.h"
#include "cora/cont_set/registry.h"
#include "polytope_testing.h"

using namespace cora;
using test::check;
using test::column;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Points: halfspaces and vertices, boundary with tolerance, equalities.
void points(const std::string &b) {
    const Polytope H = Polytope(Tensor({{1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}}),
                                vec({1.0, 1.0, 0.0, 0.0}));
    const Polytope V(Tensor({{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}}));
    for (const Polytope &P : {H, V}) {
        check(P.contains(vec({0.5, 0.5})), b + ": inside");
        check(P.contains(vec({1.0, 1.0})), b + ": a vertex");
        check(P.contains(vec({1.0, 0.5})), b + ": on an edge");
        check(!P.contains(vec({1.5, 0.5})), b + ": outside");
        check(!P.contains(vec({-0.1, 0.5})), b + ": outside, left");
    }
    check(H.contains(vec({1.0 + 1e-15, 0.5})), b + ": within the default tolerance");
    check(!H.contains(vec({1.0 + 1e-6, 0.5})), b + ": beyond the default tolerance");
    check(H.contains(vec({1.0 + 1e-6, 0.5}), "exact", 1e-5), b + ": a given tolerance");
    const Tensor many = Tensor({{0.5, 0.5}, {2.0, 0.5}, {0.0, 1.0}});
    check(H.contains(many) == false && H.contains(many.take({0, 2}, -2)),
          b + ": a cloud: all or nothing");
    for (const char *method : {"exact:venum", "exact:polymax", "approx"})
        check(H.contains(vec({0.5, 0.5}), method) && !H.contains(vec({2.0, 0.5}), method),
              b + ": " + method);
    // the vertex routes
    check(V.contains(vec({0.5, 0.5}), "exact:polymax"), b + ": vertices: polymax");
    check(Polytope(Tensor({{1.0}, {4.0}})).contains(vec({2.0})), b + ": 1D vertices");
    const Polytope eq(Tensor::zeros({0, 2}), Tensor::zeros({0}), Tensor({{1.0, 1.0}}), vec({1.0}));
    check(eq.contains(vec({0.5, 0.5})) && !eq.contains(vec({0.5, 0.6})), b + ": an equality");
    check(Polytope::inf(2).contains(vec({1e9, 0.0})), b + ": R^n holds everything");
    check(!Polytope::empty(2).contains(vec({0.0, 0.0})), b + ": the empty set holds nothing");
    const Polytope point(Tensor({{1.0, 2.0}}));
    check(point.contains(vec({1.0, 2.0})) && !point.contains(vec({1.0, 2.1})),
          b + ": a single vertex");
    check(V.contains(vec({0.5, 0.5}), "exact:venum"), b + ": vertices: venum");
}

/// Sets: a box, a zonotope, polytopes of both forms, the empty and the whole set.
void sets(const std::string &b) {
    const Polytope H = Polytope(Tensor({{1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}}),
                                vec({1.0, 1.0, 0.0, 0.0}));
    const Interval in(vec({0.2, 0.2}), vec({0.8, 0.8})), out(vec({0.5, 0.5}), vec({1.5, 1.0}));
    check(H.contains(in) && !H.contains(out), b + ": a box");
    check(H.contains(in, "approx") && !H.contains(out, "approx"), b + ": a box, approx");
    const Zonotope Z(vec({0.5, 0.5}), Tensor({{0.3, 0.0}, {0.1, 0.3}}));
    check(H.contains(Z) && H.contains(Z, "approx"), b + ": a zonotope");
    check(!H.contains(Zonotope(vec({1.0, 1.0}), Tensor({{0.3, 0.0}, {0.0, 0.3}}))),
          b + ": a zonotope out");
    const Polytope small(Tensor({{0.2, 0.2}, {0.8, 0.2}, {0.8, 0.8}}));
    check(H.contains(small) && !small.contains(H), b + ": a vertex polytope");
    check(H.contains(Polytope(Tensor({{-1.0, 0.0}, {0.0, -1.0}, {1.0, 1.0}}),
                              vec({0.0, 0.0, 1.5}))) == false,
          b + ": a halfspace polytope sticking out");
    check(
        H.contains(Polytope(Tensor({{-1.0, 0.0}, {0.0, -1.0}, {1.0, 1.0}}), vec({0.0, 0.0, 1.0}))),
        b + ": a halfspace polytope inside");
    // containment of polytopes and boxes
    check(H.contains(Polytope::empty(2)) && !Polytope::empty(2).contains(H),
          b + ": empty is inside everything");
    check(Polytope::inf(2).contains(H) && !H.contains(Polytope::inf(2)), b + ": R^n");
    check(Polytope(Tensor({{1.0}, {4.0}})).contains(Interval(vec({2.0}), vec({3.0}))),
          b + ": 1D: a box");
    check(H.contains(Interval::empty(2)), b + ": the empty box");
    check(H.contains(H) &&
              Polytope(Tensor({{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}})).contains(H),
          b + ": a set contains itself");
}

/// The table holds the pairs of a polytope with a constrained zonotope, a bundle, an ellipsoid and
/// a capsule: exact by the support functions along the rows, as for a zonotope.
void table(const std::string &b) {
    const Polytope H = Polytope(Tensor({{1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}}),
                                vec({1.0, 1.0, 0.0, 0.0}));
    const Zonotope in(vec({0.5, 0.5}), Tensor::eye(2) * 0.3),
        out(vec({1.0, 1.0}), Tensor::eye(2) * 0.3);
    check(contains(H, in) && !contains(H, out), b + ": a zonotope");
    check(contains(H, ConZonotope(in)) && !contains(H, ConZonotope(out)),
          b + ": a constrained zonotope");
    check(contains(H, ZonoBundle({in, in})) && !contains(H, ZonoBundle({out, out})),
          b + ": a bundle");
    check(contains(H, Ellipsoid(vec({0.5, 0.5}), Tensor::eye(2) * 0.01)) &&
              !contains(H, Ellipsoid(vec({1.0, 1.0}), Tensor::eye(2) * 0.01)),
          b + ": an ellipsoid");
    check(contains(H, Capsule(vec({0.5, 0.5}), vec({0.1, 0.0}), 0.1)) &&
              !contains(H, Capsule(vec({1.0, 1.0}), vec({0.1, 0.0}), 0.1)),
          b + ": a capsule");
}

/// What is refused: the method, shapes, 'approx' for other sets; checks off.
void refusals() {
    const Polytope H = Polytope(Tensor({{1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}}),
                                vec({1.0, 1.0, 0.0, 0.0}));
    check(test::raises([&] { H.contains(vec({0.0, 0.0}), "nope"); }, Error::no_specific_alg),
          "contains: an unknown method");
    check(test::raises([&] { H.contains(vec({0.0})); }, Error::dimension_mismatch),
          "contains: the point has the dimension of the set");
    check(test::raises([&] { H.contains(Interval(vec({0.0}), vec({1.0}))); },
                       Error::dimension_mismatch),
          "contains: the set has the dimension of the polytope");
    check(test::raises([&] { H.contains(H, "approx"); }, Error::not_supported),
          "contains: approx needs a zonotope or a box");
    check(test::raises(
              [&] {
                  Polytope(Tensor::ones({2, 1, 2}), Tensor::ones({2, 1})).contains(vec({0.0, 0.0}));
              },
              Error::not_supported),
          "contains: a batch of polytopes");
    ChecksScope off(false);
    // the same with the checks off
    check(H.contains(vec({0.5, 0.5})), "contains: the same with the checks off");
}

/// Samples are contained, probes beyond the box are not, the vertex route agrees with the
/// halfspace route, and a polytope contains the boxes of its inner scaling.
void properties(const std::string &b) {
    Rng rng(4);
    for (const int n : {2, 3}) {
        const Polytope Q = test::random_polytope(rng, n, 2 * n);
        check(Q.contains(test::sample_points(Q, rng, 30)),
              b + ": samples, " + std::to_string(n) + "D");
        check(!Q.contains(Q.interval().sup() + 0.5),
              b + ": beyond the box, " + std::to_string(n) + "D");
        const Tensor points = test::sphere_points(rng, n, n + 3);
        const Polytope V(points), H(V.constraints().A, V.constraints().b);
        bool same = true;
        for (int k = 0; k < 30; ++k) {
            std::vector<double> x(static_cast<std::size_t>(n));
            rng.uniform(x.data(), x.size(), -1.2, 1.2);
            const Tensor p = vec(x);
            same = same && V.contains(p) == H.contains(p);
        }
        check(same, b + ": vertices and halfspaces agree, " + std::to_string(n) + "D");
        const Interval inner(Q.center() - 0.05, Q.center() + 0.05);
        // a small box about the center
        check(Q.contains(inner), b + ": a small box about the center, " + std::to_string(n) + "D");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        properties(b);
        points(b);
        sets(b);
        table(b);
    });
    refusals();
    return test::finish("polytope contains");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
