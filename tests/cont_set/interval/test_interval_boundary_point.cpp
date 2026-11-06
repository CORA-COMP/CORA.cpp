// test_interval_boundary_point - interval boundary_point: where a ray leaves the box
//
// CORA: contSet/@interval/boundaryPoint.m
// PROPERTY: contSet/@interval/boundaryPoint.m
// CORA authors: Mark Wetzlinger
// Authors:       Tobias Ladner, Claude Sonnet 5.5, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <cmath>
#include <limits>

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

const double kInf = std::numeric_limits<double>::infinity();

/// The ray along (1, 1) from the center (2.5, 1) of [1,4] x [-1,3] leaves through the right face.
void boundary_point_from_the_center_and_from_a_start(const std::string &b) {
    const Interval I(column({1.0, -1.0}), column({4.0, 3.0}));
    check(close(I.boundary_point(column({1.0, 1.0})), std::vector<double>{4.0, 2.5}),
          b + ": from the center");
    check(close(I.boundary_point(column({1.0, 0.0}), column({2.0, 0.0})),
                std::vector<double>{4.0, 0.0}),
          b + ": from a start point, along an axis");
    check(close(I.boundary_point(column({-2.0, -1.0}), column({3.0, 2.0})),
                std::vector<double>{1.0, 1.0}),
          b + ": a negative direction leaves through the left face");
    check(close(I.boundary_point(column({2.0, 2.0}), column({2.0, 0.0})),
                I.boundary_point(column({1.0, 1.0}), column({2.0, 0.0}))),
          b + ": the length of dir does not matter");
}

/// A degenerate box is left at once; an empty box has no boundary point.
void boundary_point_of_special_boxes(const std::string &b) {
    const Interval point(column({1.0, 2.0}), column({1.0, 2.0}));
    check(close(point.boundary_point(column({1.0, 1.0})), std::vector<double>{1.0, 2.0}),
          b + ": a point is its own boundary point");
    const Tensor none = point.inf().zeros_like({2, 0});
    check(Interval(none, none).boundary_point(column({1.0, 1.0}), column({0.0, 0.0})).shape() ==
              std::vector<int64_t>({2, 0}),
          b + ": an empty box gives (n, 0)");
}

/// A box unbounded on one side is left through the finite one; where no bound is finite the ray
/// ends at infinity, in the coordinates of dir (CORA returns -Inf for [0, Inf] and dir -1, start
/// 1).
void boundary_point_of_unbounded_boxes(const std::string &b) {
    const Interval half(column({0.0, 0.0}), column({kInf, kInf}));
    check(close(half.boundary_point(column({-1.0, 0.0}), column({1.0, 2.0})),
                std::vector<double>{0.0, 2.0}),
          b + ": left through the finite lower bound");
    const Tensor ends = half.boundary_point(column({1.0, 0.0}), column({1.0, 2.0}));
    check(ends.data() == std::vector<double>({kInf, 2.0}), b + ": a ray that never leaves");
    check(Interval::unbounded(2).boundary_point(column({1.0, -1.0})).data() ==
              std::vector<double>({kInf, -kInf}),
          b + ": the whole space, from its center (the origin)");
}

/// The result lies on the boundary of the box, on the ray, and just beyond it is outside.
void boundary_point_is_the_exit_of_the_ray(const std::string &b) {
    cora::Rng rng(21);
    for (int trial = 0; trial < 30; ++trial) {
        const Interval I = Interval::generate_random(rng, 4);
        const Tensor start = I.rand_point(1, rng), dir = column(test::random_direction(rng, 4));
        const Tensor x = I.boundary_point(dir, start);
        check(I.contains(x), b + ": the exit point is in the box");
        const Tensor beyond = x + (x - start) * 1e-3;
        check(!I.contains(beyond, 0), b + ": beyond the exit point is outside");
        // x - start is a positive multiple of dir
        const std::vector<double> d = dir.data(), s = (x - start).data();
        bool parallel = true;
        for (std::size_t i = 0; i < d.size(); ++i)
            parallel = parallel && std::abs(s[i] * d[0] - s[0] * d[i]) < 1e-9;
        check(parallel && s[0] * d[0] >= 0, b + ": the exit point is on the ray");
    }
}

/// A zero direction, a start point outside, an unbounded box without start and wrong shapes are
/// refused; with the checks off the exit point is the same.
void checks_and_their_switch() {
    const Interval I(column({1.0, -1.0}), column({4.0, 3.0}));
    check(test::raises([&] { I.boundary_point(column({0.0, 0.0})); }, Error::wrong_value),
          "boundary_point: a zero direction");
    check(test::raises([&] { I.boundary_point(column({1.0, 1.0}), column({9.0, 0.0})); },
                       Error::wrong_value),
          "boundary_point: a start point outside the box");
    check(test::raises(
              [&] {
                  Interval(column({0.0, 0.0}), column({kInf, 1.0}))
                      .boundary_point(column({1.0, 1.0}));
              },
              Error::wrong_value),
          "boundary_point: a box with an infinite center needs a start point");
    check(
        test::raises([&] { I.boundary_point(column({1.0, 1.0, 1.0})); }, Error::dimension_mismatch),
        "boundary_point: dir of dimension 3");
    check(test::raises([&] { I.boundary_point(column({1.0, 1.0}), column({2.0})); },
                       Error::dimension_mismatch),
          "boundary_point: start of dimension 1");
    const Tensor on = I.boundary_point(column({1.0, 1.0}));
    ChecksScope off(false);
    check(close(I.boundary_point(column({1.0, 1.0})), on, 0),
          "boundary_point: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        boundary_point_from_the_center_and_from_a_start(b);
        boundary_point_of_special_boxes(b);
        boundary_point_of_unbounded_boxes(b);
        boundary_point_is_the_exit_of_the_ray(b);
    });
    checks_and_their_switch();
    return test::finish("interval boundary_point");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
