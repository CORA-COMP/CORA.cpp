// test_polytope_contains - Polytope::contains: points and sets
//
// CORA: contSet/@polytope/contains_.m
// PROPERTY: the points of the set are contained and points outside are not; sets are contained
//           iff their support function stays below every row

#include "polytope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// Points: halfspaces and vertices, boundary with tolerance, equalities.
void points(const std::string &b) {
    const Polytope H = Polytope(Tensor({{1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}}),
                                column({1.0, 1.0, 0.0, 0.0}));
    const Polytope V(Tensor({{0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}}));
    for (const Polytope &P : {H, V}) {
        check(P.contains(column({0.5, 0.5})), b + ": inside");
        check(P.contains(column({1.0, 1.0})), b + ": a vertex");
        check(P.contains(column({1.0, 0.5})), b + ": on an edge");
        check(!P.contains(column({1.5, 0.5})), b + ": outside");
        check(!P.contains(column({-0.1, 0.5})), b + ": outside, left");
    }
    check(H.contains(column({1.0 + 1e-15, 0.5})), b + ": within the default tolerance");
    check(!H.contains(column({1.0 + 1e-6, 0.5})), b + ": beyond the default tolerance");
    check(H.contains(column({1.0 + 1e-6, 0.5}), "exact", 1e-5), b + ": a given tolerance");
    const Tensor many = Tensor({{0.5, 2.0, 0.0}, {0.5, 0.5, 1.0}});
    check(H.contains(many) == false && H.contains(many.take({0, 2}, -1)),
          b + ": a cloud: all or nothing");
    for (const char *method : {"exact:venum", "exact:polymax", "approx"})
        check(H.contains(column({0.5, 0.5}), method) && !H.contains(column({2.0, 0.5}), method),
              b + ": " + method);
    check(V.contains(column({0.5, 0.5}), "exact:polymax"), b + ": vertices: polymax");
    check(Polytope(Tensor({{1.0, 4.0}})).contains(column({2.0})), b + ": 1D vertices");
    const Polytope eq(Tensor::zeros({0, 2}), Tensor::zeros({0, 1}), Tensor({{1.0, 1.0}}),
                      column({1.0}));
    check(eq.contains(column({0.5, 0.5})) && !eq.contains(column({0.5, 0.6})), b + ": an equality");
    check(Polytope::inf(2).contains(column({1e9, 0.0})), b + ": R^n holds everything");
    check(!Polytope::empty(2).contains(column({0.0, 0.0})), b + ": the empty set holds nothing");
    const Polytope point(column({1.0, 2.0}));
    check(point.contains(column({1.0, 2.0})) && !point.contains(column({1.0, 2.1})),
          b + ": a single vertex");
    check(V.contains(column({0.5, 0.5}), "exact:venum"), b + ": vertices: venum");
}

/// Sets: a box, a zonotope, polytopes of both forms, the empty and the whole set.
void sets(const std::string &b) {
    const Polytope H = Polytope(Tensor({{1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}}),
                                column({1.0, 1.0, 0.0, 0.0}));
    const Interval in(column({0.2, 0.2}), column({0.8, 0.8})),
        out(column({0.5, 0.5}), column({1.5, 1.0}));
    check(H.contains(in) && !H.contains(out), b + ": a box");
    check(H.contains(in, "approx") && !H.contains(out, "approx"), b + ": a box, approx");
    const Zonotope Z(column({0.5, 0.5}), Tensor({{0.3, 0.1}, {0.0, 0.3}}));
    check(H.contains(Z) && H.contains(Z, "approx"), b + ": a zonotope");
    check(!H.contains(Zonotope(column({1.0, 1.0}), Tensor({{0.3, 0.0}, {0.0, 0.3}}))),
          b + ": a zonotope out");
    const Polytope small(Tensor({{0.2, 0.8, 0.8}, {0.2, 0.2, 0.8}}));
    check(H.contains(small) && !small.contains(H), b + ": a vertex polytope");
    check(H.contains(Polytope(Tensor({{-1.0, 0.0}, {0.0, -1.0}, {1.0, 1.0}}),
                              column({0.0, 0.0, 1.5}))) == false,
          b + ": a halfspace polytope sticking out");
    check(H.contains(
              Polytope(Tensor({{-1.0, 0.0}, {0.0, -1.0}, {1.0, 1.0}}), column({0.0, 0.0, 1.0}))),
          b + ": a halfspace polytope inside");
    check(H.contains(Polytope::empty(2)) && !Polytope::empty(2).contains(H),
          b + ": empty is inside everything");
    check(Polytope::inf(2).contains(H) && !H.contains(Polytope::inf(2)), b + ": R^n");
    check(Polytope(Tensor({{1.0, 4.0}})).contains(Interval(column({2.0}), column({3.0}))),
          b + ": 1D: a box");
    check(H.contains(Interval::empty(2)), b + ": the empty box");
    check(H.contains(H) &&
              Polytope(Tensor({{0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}})).contains(H),
          b + ": a set contains itself");
}

/// What is refused: the method, shapes, 'approx' for other sets; checks off.
void refusals() {
    const Polytope H = Polytope(Tensor({{1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}}),
                                column({1.0, 1.0, 0.0, 0.0}));
    check(test::raises([&] { H.contains(column({0.0, 0.0}), "nope"); }, Error::no_specific_alg),
          "contains: an unknown method");
    check(test::raises([&] { H.contains(column({0.0})); }, Error::dimension_mismatch),
          "contains: the point has the dimension of the set");
    check(test::raises([&] { H.contains(Interval(column({0.0}), column({1.0}))); },
                       Error::dimension_mismatch),
          "contains: the set has the dimension of the polytope");
    check(test::raises([&] { H.contains(H, "approx"); }, Error::not_supported),
          "contains: approx needs a zonotope or a box");
    check(test::raises(
              [&] {
                  Polytope(Tensor::ones({2, 1, 2}), Tensor::ones({2, 1, 1}))
                      .contains(column({0.0, 0.0}));
              },
              Error::not_supported),
          "contains: a batch of polytopes");
    ChecksScope off(false);
    check(H.contains(column({0.5, 0.5})), "contains: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        points(b);
        sets(b);
    });
    refusals();
    return test::finish("polytope contains");
}
