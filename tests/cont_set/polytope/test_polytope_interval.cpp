// test_polytope_interval - Polytope::interval: the tightest box around the set
//
// CORA: contSet/@polytope/interval.m
// PROPERTY: the box of vertices is their coordinate-wise extremes; of halfspaces it is the 2n
//           linear programs, whose bounds are met by points of the set

#include "polytope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// Vertices: the extremes; a batch works; the empty set is the empty box.
void vertices(const std::string &b) {
    const Interval I = Polytope(Tensor({{0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}})).interval();
    check(close(I.inf(), {0, 0}, 0) && close(I.sup(), {1, 1}, 0), b + ": the square");
    check(Polytope::empty(2).interval().is_empty(), b + ": the empty set");
    const Interval batch = Polytope(Tensor::ones({3, 2, 4})).interval();
    check(batch.inf().shape() == Shape{3, 2, 1}, b + ": a batch of three");
}

/// Halfspaces: the linear programs; unbounded sides are infinite; an infeasible set is empty.
void halfspaces(const std::string &b) {
    const Polytope tri(Tensor({{-1.0, 0.0}, {0.0, -1.0}, {1.0, 1.0}}), column({0.0, 0.0, 2.0}));
    const Interval I = tri.interval();
    check(close(I.inf(), {0, 0}, 1e-9) && close(I.sup(), {2, 2}, 1e-9), b + ": a triangle");
    const double inf = std::numeric_limits<double>::infinity();
    const Interval half = Polytope(Tensor({{1.0, 0.0}}), column({1.0})).interval();
    check(half.inf().data()[0] == -inf && half.sup().data()[0] == 1.0 &&
              half.sup().data()[1] == inf,
          b + ": a halfspace: unbounded sides");
    const Polytope none(Tensor({{1.0, 0.0}, {-1.0, 0.0}}), column({0.0, -1.0}));
    check(none.interval().is_empty(), b + ": infeasible: the empty box");
    check(Polytope::inf(2).interval().sup().min().item() == inf, b + ": R^2");
    const Polytope eq(Tensor({{1.0, 0.0}, {-1.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}}),
                      column({2.0, 2.0, 2.0, 2.0}), Tensor({{1.0, 1.0}}), column({1.0}));
    check(close(eq.interval().inf(), {-1, -1}, 1e-9) && close(eq.interval().sup(), {2, 2}, 1e-9),
          b + ": with an equality");
}

/// A checks-off call on a valid set.
void refusals() {
    const Polytope P(Tensor({{0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}}));
    ChecksScope off(false);
    check(close(P.interval().sup(), {1, 1}, 0), "interval: the same with the checks off");
    check(
        test::raises([&] { Polytope(Tensor::ones({2, 1, 2}), Tensor::ones({2, 1, 1})).interval(); },
                     Error::not_supported),
        "interval: a batch of halfspace polytopes");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        vertices(b);
        halfspaces(b);
    });
    refusals();
    return test::finish("polytope interval");
}
