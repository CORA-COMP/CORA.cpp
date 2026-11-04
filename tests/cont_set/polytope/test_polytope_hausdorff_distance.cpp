// test_polytope_hausdorff_distance - Polytope::hausdorff_distance: the Hausdorff distance
//
// CORA: contSet/@polytope/hausdorffDist.m
// PROPERTY: the distance is symmetric, zero for equal sets, and the largest distance of a vertex
//           of one set to the other set

#include "polytope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// Hand-computed distances between squares, a point, boxes and halfspace sets.
void distances(const std::string &b) {
    const Polytope S1(Tensor({{0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}}));
    const Polytope S2(Tensor({{2.0, 3.0, 3.0, 2.0}, {0.0, 0.0, 1.0, 1.0}}));
    check(close(S1.hausdorff_distance(S2), 2.0, 1e-8), b + ": two squares");
    check(close(S2.hausdorff_distance(S1), 2.0, 1e-8), b + ": symmetric");
    check(close(S1.hausdorff_distance(S1), 0.0, 1e-8), b + ": equal sets");
    const Polytope big(Tensor({{0.0, 2.0, 2.0, 0.0}, {0.0, 0.0, 2.0, 2.0}}));
    check(close(S1.hausdorff_distance(big), std::sqrt(2.0), 1e-8), b + ": nested squares");
    check(close(S1.hausdorff_distance(Tensor({{3.0, 0.0}, {0.0, 0.0}})), 2.0, 1e-8),
          b + ": the farthest point");
    check(close(S1.hausdorff_distance(Interval(column({0.0, 0.0}), column({2.0, 2.0}))),
                std::sqrt(2.0), 1e-8),
          b + ": a box");
    const Polytope H(Tensor({{1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}}),
                     column({1.0, 1.0, 0.0, 0.0}));
    check(close(H.hausdorff_distance(S2), 2.0, 1e-8) && close(H.hausdorff_distance(S1), 0.0, 1e-8),
          b + ": halfspaces");
    check(close(Polytope(Tensor({{0.0, 1.0}})).hausdorff_distance(Polytope(Tensor({{3.0, 5.0}}))),
                4.0, 1e-8),
          b + ": 1D");
    const Polytope inf = Polytope::inf(2);
    check(close(inf.hausdorff_distance(inf), 0.0, 0.0), b + ": R^n and itself");
    const double d = Polytope::empty(2).hausdorff_distance(S1);
    check(std::isinf(d), b + ": the empty set and a set: infinite");
    check(close(Polytope::empty(2).hausdorff_distance(Polytope::empty(2)), 0.0, 0.0),
          b + ": two empty sets");
    check(std::isinf(inf.hausdorff_distance(S1)), b + ": R^n and a bounded set");
}

/// What is refused; checks off.
void refusals() {
    const Polytope S(Tensor({{0.0, 1.0, 1.0}, {0.0, 0.0, 1.0}}));
    check(test::raises([&] { S.hausdorff_distance(Polytope::inf(3)); }, Error::dimension_mismatch),
          "hausdorff_distance: dimensions");
    check(test::raises([&] { S.hausdorff_distance(Tensor::zeros({3, 1})); },
                       Error::dimension_mismatch),
          "hausdorff_distance: points have the dimension");
    check(test::raises([&] { S.hausdorff_distance(Interval::unbounded(3)); },
                       Error::dimension_mismatch),
          "hausdorff_distance: the box has the dimension");
    ChecksScope off(false);
    check(close(S.hausdorff_distance(S), 0.0, 1e-8),
          "hausdorff_distance: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) { distances(b); });
    refusals();
    return test::finish("polytope hausdorff_distance");
}
