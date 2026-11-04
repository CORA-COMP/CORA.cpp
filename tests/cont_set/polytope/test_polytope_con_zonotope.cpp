// test_polytope_con_zonotope - Polytope::con_zonotope: the polytope as a constrained zonotope
//
// CORA: contSet/@polytope/conZonotope.m
// PROPERTY: the constrained zonotope holds the points of the polytope and no others; it is the
//           same set whichever method and box are used

#include "polytope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// Whether the constrained zonotope and the polytope hold the same points of a probe set.
bool same_set(const ConZonotope &cz, const Polytope &P, Rng &rng) {
    for (int k = 0; k < 40; ++k) {
        std::vector<double> x(static_cast<std::size_t>(P.dim()));
        rng.uniform(x.data(), x.size(), -1.5, 2.5);
        const Tensor p = column(x);
        if (cz.contains(p) != P.contains(p, "exact", 1e-9)) return false;
    }
    return true;
}

/// Both methods, and a given box, on a triangle given by halfspaces and one by vertices.
void methods(const std::string &b) {
    Rng rng(3);
    const Polytope tri(Tensor({{-1.0, 0.0}, {0.0, -1.0}, {1.0, 1.0}}), column({0.0, 0.0, 2.0}));
    for (const char *method : {"exact:supportFunc", "exact:vertices"})
        check(same_set(tri.con_zonotope(method), tri, rng), b + ": " + method + ": the triangle");
    const Interval wide(column({-1.0, -1.0}), column({3.0, 3.0}));
    check(same_set(tri.con_zonotope("exact:supportFunc", wide), tri, rng), b + ": a wider box");
    const Polytope sq(Tensor({{0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}}));
    for (const char *method : {"exact:supportFunc", "exact:vertices"})
        check(same_set(sq.con_zonotope(method), sq, rng), b + ": " + method + ": vertices");
    const Polytope eq(Tensor({{1.0, 0.0}, {-1.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}}),
                      column({2.0, 2.0, 2.0, 2.0}), Tensor({{1.0, 1.0}}), column({1.0}));
    check(same_set(eq.con_zonotope(), eq, rng), b + ": with an equality");
    check(same_set(eq.con_zonotope("exact:vertices"), eq, rng), b + ": vertices: with an equality");
    const ConZonotope cz = tri.con_zonotope();
    check(close(cz.c(), {1, 1}, 1e-9) && cz.generators().shape() == Shape{2, 5},
          b + ": center, generators");
}

/// The special sets: empty, R^n, unbounded, a point.
void special(const std::string &b) {
    check(!Polytope::empty(2).con_zonotope().contains(column({0.0, 0.0})), b + ": the empty set");
    check(close(Polytope(column({1.0, 2.0})).con_zonotope().c(), {1, 2}, 1e-9), b + ": a point");
}

/// What is refused.
void refusals() {
    const Polytope sq(Tensor({{0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}}));
    check(test::raises([&] { sq.con_zonotope("nope"); }, Error::wrong_value),
          "con_zonotope: method");
    check(test::raises([&] { sq.con_zonotope("exact:supportFunc", Interval::unbounded(3)); },
                       Error::dimension_mismatch),
          "con_zonotope: the box has the dimension of the set");
    check(test::raises([&] { Polytope::inf(2).con_zonotope(); }, Error::special_error),
          "con_zonotope: R^n is unbounded");
    const Polytope open(Tensor({{1.0, 0.0}}), column({1.0}));
    check(test::raises([&] { open.con_zonotope(); }, Error::special_error),
          "con_zonotope: a halfspace is unbounded");
    check(test::raises([&] { open.con_zonotope("exact:vertices"); }, Error::special_error),
          "con_zonotope: vertices of an unbounded set");
    ChecksScope off(false);
    check(close(sq.con_zonotope().c(), {0.5, 0.5}, 1e-9),
          "con_zonotope: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        methods(b);
        special(b);
    });
    refusals();
    return test::finish("polytope con_zonotope");
}
