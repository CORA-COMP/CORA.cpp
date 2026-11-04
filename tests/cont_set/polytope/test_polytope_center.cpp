// test_polytope_center - Polytope::center: the Chebyshev center or the mean of the vertices
//
// CORA: contSet/@polytope/center.m
// PROPERTY: the Chebyshev center is the center of the largest ball (its distance to every facet
//           is the radius); 'avg' is the mean of the extreme vertices

#include "polytope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// Both methods on the unit square and on a triangle.
void methods(const std::string &b) {
    const Polytope sq(Tensor({{0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}}));
    check(close(sq.center("avg"), {0.5, 0.5}), b + ": avg: the square");
    check(close(sq.center("chebyshev"), {0.5, 0.5}), b + ": chebyshev: the square");
    check(close(sq.center(), {0.5, 0.5}), b + ": the default is chebyshev");
    // the mean counts the extreme vertices only; the middle of an edge is none
    const Polytope with_middle(Tensor({{0.0, 1.0, 0.0, 0.5}, {0.0, 0.0, 1.0, 0.0}}));
    check(close(with_middle.center("avg"), {1.0 / 3.0, 1.0 / 3.0}),
          b + ": avg ignores inner points");
    const Polytope tri(Tensor({{-1.0, 0.0}, {0.0, -1.0}, {1.0, 1.0}}), column({0.0, 0.0, 2.0}));
    const double r = 2.0 / (2.0 + std::sqrt(2.0));
    check(close(tri.center("chebyshev"), {r, r}, 1e-8),
          b + ": chebyshev: the incircle of a triangle");
    check(close(tri.center("avg"), {2.0 / 3.0, 2.0 / 3.0}, 1e-9), b + ": avg of the halfspace set");
}

/// The special sets.
void special(const std::string &b) {
    check(close(Polytope::inf(2).center(), {0, 0}, 0), b + ": R^n: the origin");
    check(Polytope::empty(2).center().shape() == Shape{2, 0}, b + ": the empty set: no center");
    check(Polytope::empty(2).center("avg").shape() == Shape{2, 0}, b + ": avg: no center");
    check(close(Polytope(column({1.0, 2.0})).center(), {1, 2}), b + ": a point");
    check(close(Polytope(Tensor({{1.0, 3.0}})).center(), {2}), b + ": 1D");
    const Polytope eq(Tensor::zeros({0, 2}), Tensor::zeros({0, 1}),
                      Tensor({{1.0, 0.0}, {0.0, 1.0}}), column({1.0, 2.0}));
    check(close(eq.center(), {1, 2}), b + ": only equalities: the point");
    const Polytope ineq(Tensor({{1.0, 1.0}, {-1.0, -1.0}}), column({1.0, 1.0}),
                        Tensor({{1.0, -1.0}}), column({0.0}));
    check(close(ineq.center(), {0, 0}, 1e-8), b + ": a segment given by rows");
    const Polytope conflict(Tensor::zeros({0, 1}), Tensor::zeros({0, 1}), Tensor({{1.0}, {1.0}}),
                            column({0.0, 1.0}));
    check(conflict.center().shape() == Shape{1, 0}, b + ": contradicting equalities: empty");
    const Polytope consistent(Tensor::zeros({0, 2}), Tensor::zeros({0, 1}),
                              Tensor({{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}}),
                              column({1.0, 2.0, 3.0}));
    check(close(consistent.center(), {1, 2}), b + ": more equalities than dimensions, consistent");
}

/// The method is checked; unbounded sets have no Chebyshev ball; checks off.
void refusals() {
    const Polytope P(Tensor({{0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}}));
    check(test::raises([&] { P.center("mean"); }, Error::wrong_value), "center: an unknown method");
    ChecksScope off(false);
    check(close(P.center("avg"), {0.5, 0.5}), "center: the same with the checks off");
    check(test::raises([&] { Polytope(Tensor::ones({2, 1, 2}), Tensor::ones({2, 1, 1})).center(); },
                       Error::not_supported),
          "center: a batch of halfspace polytopes");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        methods(b);
        special(b);
    });
    refusals();
    return test::finish("polytope center");
}
