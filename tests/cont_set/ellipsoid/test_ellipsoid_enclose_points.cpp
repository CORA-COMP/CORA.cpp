// test_ellipsoid_enclose_points - Ellipsoid::enclose_points: an ellipsoid around a point cloud
//
// CORA: contSet/@ellipsoid/enclosePoints.m
// PROPERTY: contSet/@ellipsoid/enclosePoints.m
// CORA authors: Niklas Kochdumper, Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "ellipsoid_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: a line of points is its range, a single point is a point, a symmetric cloud
/// is centered at its mean.
void enclose_points_known(const std::string &b) {
    // two points on the x axis: the segment [1, 3], radius 1 around 2
    const Ellipsoid segment = Ellipsoid::enclose_points(Tensor({{1.0, 3.0}, {0.0, 0.0}}));
    check(close(segment.q(), {2, 0}, 1e-12), b + ": the center of a segment");
    check(close(segment.Q().abs(), {1, 0, 0, 0}, 1e-12) && !segment.is_full_dim(),
          b + ": a degenerate cloud: the range, Q = diag(1, 0)");
    // one point: that point (CORA raises an error)
    const Ellipsoid point = Ellipsoid::enclose_points(Tensor({{2.0, 2.0}, {3.0, 3.0}}));
    check(close(point.q(), {2, 3}, 0) && close(point.Q(), Tensor::zeros({2, 2}), 0),
          b + ": coinciding points give the point");
    check(close(Ellipsoid::enclose_points(column({4.0, 5.0})).q(), {4, 5}, 0), b + ": one column");
    // the corners of a square: center 0, every corner inside, the farthest on the boundary
    const Tensor corners({{1.0, 1.0, -1.0, -1.0}, {1.0, -1.0, 1.0, -1.0}});
    const Ellipsoid square = Ellipsoid::enclose_points(corners);
    check(close(square.q(), {0, 0}, 1e-12) && test::all_inside(square, corners),
          b + ": the corners of a square");
    check(test::close(Tensor({{test::norms_of(square, corners)[0]}}), Tensor({{1.0}}), 1e-9),
          b + ": the farthest point is on the boundary");
    // one dimension: the range
    const Ellipsoid line = Ellipsoid::enclose_points(Tensor({{1.0, 2.0, 7.0}}));
    check(close(line.q(), {4}, 1e-12) && close(line.Q(), {9}, 1e-12), b + ": one dimension");
}

/// Stated property: every point is in the ellipsoid, also for fewer points than dimensions and for
/// clouds in a subspace.
void enclose_points_property(const std::string &b) {
    cora::Rng rng(13);
    for (const int64_t count : {2, 3, 5, 40}) {
        const Tensor P = test::normal_matrix(rng, 4, count);
        check(test::all_inside(Ellipsoid::enclose_points(P), P), b + ": a random cloud");
    }
    // a cloud in a plane of R^4
    const Tensor plane = test::normal_matrix(rng, 4, 2).matmul(test::normal_matrix(rng, 2, 30));
    const Ellipsoid E = Ellipsoid::enclose_points(plane);
    check(test::all_inside(E, plane) && !E.is_full_dim(), b + ": a cloud in a plane");
    check(close(E.q(), plane.mean(1, true), 1e-12), b + ": the center is the mean");
}

/// An empty cloud, a wrong method and the SDP method are refused; with the checks off the result
/// is the same.
void checks_and_their_switch() {
    check(
        test::raises([] { Ellipsoid::enclose_points(Tensor::zeros({2, 0})); }, Error::wrong_value),
        "enclose_points: at least one point");
    check(test::raises([] { Ellipsoid::enclose_points(Tensor({{1.0, 2.0}}), "mean"); },
                       Error::wrong_value),
          "enclose_points: the method is cov or min-vol");
    const Tensor P({{1.0, 2.0, 0.0}, {0.0, 1.0, 3.0}});
    check(test::raises([&] { Ellipsoid::enclose_points(P, "min-vol"); }, Error::not_supported),
          "enclose_points: min-vol needs an SDP solver");
    check(Ellipsoid::enclose_points(Tensor({{1.0, 3.0}}), "min-vol").Q().item() == 1.0,
          "enclose_points: min-vol of one dimension needs no solver, as in CORA");
    const Ellipsoid on = Ellipsoid::enclose_points(P);
    ChecksScope off(false);
    check(close(Ellipsoid::enclose_points(P).Q(), on.Q(), 0),
          "enclose_points: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        enclose_points_known(b);
        enclose_points_property(b);
    });
    checks_and_their_switch();
    return test::finish("ellipsoid enclose_points");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
