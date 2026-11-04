// test_capsule_enclose_points - Capsule enclose_points: a capsule around a point cloud
//
// CORA: contSet/@capsule/enclosePoints.m
// PROPERTY: contSet/@capsule/enclosePoints.m

#include "capsule_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// Hand-computed: collinear points give a segment; one point a point; none the empty capsule.
void enclose_points_known(const std::string &b) {
    // (0,0), (1,1), (2,2): the mean (1,1), the first of the farthest points (0,0), radius 0
    const Capsule line = Capsule::enclose_points(Tensor({{0.0, 1.0, 2.0}, {0.0, 1.0, 2.0}}));
    check(close(line.c(), {1, 1}, 1e-15) && close(line.generator(), {-1, -1}, 1e-15) &&
              close(line.r(), {0}, 1e-15),
          b + ": collinear points");
    const Capsule one = Capsule::enclose_points(column({1.0, 2.0}));
    check(close(one.c(), {1, 2}, 0) && close(one.generator(), {0, 0}, 0) && close(one.r(), {0}, 0),
          b + ": one point");
    check(Capsule::enclose_points(Tensor::zeros({2, 0})).is_empty(), b + ": no points");
    // on a line (n = 1) both ends are covered: CORA's maximum as the generator misses the minimum
    const Capsule interval = Capsule::enclose_points(Tensor({{0.0, 1.0, 1.1}}));
    check(interval.contains(Tensor({{0.0, 1.0, 1.1}})), b + ": in R^1 the minimum is covered");
    // a triangle: the farthest point from the mean (1, 1/3) is (1, 1) ... the first of them
    const Capsule triangle = Capsule::enclose_points(Tensor({{0.0, 2.0, 1.0}, {0.0, 0.0, 1.0}}));
    check(close(triangle.c(), {1, 1.0 / 3.0}, 1e-14), b + ": the mean");
    check(close(triangle.generator(), {-1, -1.0 / 3.0}, 1e-14),
          b + ": to the first farthest point");
    // a batch of clouds
    const Tensor clouds = Tensor::from_data({0, 2, 0, 0, 1, 3, 0, 0}, {2, 2, 2});
    const Capsule batch = Capsule::enclose_points(clouds);
    check(batch.c().shape() == Shape{2, 2, 1} && close(batch.c(), {1, 0, 2, 0}, 0),
          b + ": a batch of clouds");
}

/// Stated property: every point is in the capsule, and the radius is attained (no smaller one for
/// this axis encloses the points).
void enclose_points_property(const std::string &b) {
    Rng rng(29);
    for (int k = 0; k < 8; ++k) {
        std::vector<double> data(3 * 12);
        rng.normal(data.data(), data.size(), 1.0);
        const Tensor points = Tensor::from_data(data, {3, 12});
        const Capsule C = Capsule::enclose_points(points);
        check(C.contains(points), b + ": all points are inside");
        const Capsule smaller(C.c(), C.generator(), C.r().data()[0] * 0.999);
        check(!smaller.contains(points), b + ": the radius is the largest distance");
    }
}

/// A scalar is no point matrix; with the checks off the same valid call works.
void checks_and_their_switch() {
    check(test::raises([] { Capsule::enclose_points(Tensor()); }, Error::wrong_value),
          "enclose_points: a matrix of points");
    check(test::raises([] { Capsule::enclose_points(Tensor::zeros({3})); }, Error::wrong_value),
          "enclose_points: columns, not a vector without rows");
    const Tensor points({{0.0, 1.0, 2.0}, {0.0, 1.0, 2.0}});
    const Capsule on = Capsule::enclose_points(points);
    ChecksScope off(false);
    check(close(Capsule::enclose_points(points).c(), on.c(), 0),
          "enclose_points: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        enclose_points_known(b);
        enclose_points_property(b);
    });
    checks_and_their_switch();
    return test::finish("capsule enclose_points");
}
