// test_zonotope_enclose_points - zonotope enclose_points: a zonotope around a cloud of points
//
// CORA: contSet/@zonotope/enclosePoints.m
// PROPERTY: contSet/@zonotope/enclosePoints.m
// CORA authors: Niklas Kochdumper
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A cloud of N points in n dimensions, correlated: a random linear image of a normal cloud.
Tensor cloud(cora::Rng &rng, int n, int N) {
    std::vector<double> z(n * N), a(n * n);
    rng.normal(z.data(), z.size(), 1.0);
    rng.normal(a.data(), a.size(), 1.0);
    return Tensor::from_data(a, {n, n}).matmul(Tensor::from_data(z, {n, int64_t(N)})) +
           column(std::vector<double>(n, 2.0));
}

/// Whether every column of the points is in the zonotope.
bool contains_all(const Zonotope &Z, const Tensor &points) {
    return Z.contains(points, "exact", 1e-9);
}

/// Hand-computed: the frame U = I gives the box around the points.
void enclose_points_known(const std::string &b) {
    const Tensor points({{0.0, 2.0, 2.0}, {0.0, 2.0, 0.0}});
    const Zonotope Z = Zonotope::enclose_points(points, Tensor::eye(2));
    check(close(Z.center(), std::vector<double>{1, 1}, 1e-14),
          b + ": frame I: the center of the box [0, 2]^2");
    check(close(Z.generators(), std::vector<double>{1, 0, 0, 1}, 1e-14),
          b + ": frame I: the radii");
    const double s = std::sqrt(0.5);
    const Zonotope R = Zonotope::enclose_points(points, Tensor({{s, -s}, {s, s}}));
    check(contains_all(R, points), b + ": a rotated frame contains the points");
    const Zonotope one = Zonotope::enclose_points(column({1.0, 2.0}), "stursberg");
    check(close(one.center(), std::vector<double>{1, 2}, 1e-14) &&
              one.generators().abs().max().item() == 0,
          b + ": stursberg: one point is a point");
    const Zonotope same = Zonotope::enclose_points(Tensor({{1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}}));
    check(close(same.center(), std::vector<double>{1, 2}, 1e-14) &&
              same.generators().abs().max().item() == 0,
          b + ": maiga: equal points give a point, where CORA divides 0 / 0");
}

/// Stated property: both methods contain every point, in 2, 3 and 4 dimensions.
void enclose_points_contain_the_cloud(const std::string &b) {
    cora::Rng rng(40);
    for (const int n : {2, 3, 4}) {
        const Tensor points = cloud(rng, n, 60);
        for (const char *method : {"maiga", "stursberg"}) {
            const Zonotope Z = Zonotope::enclose_points(points, method);
            check(Z.dim() == n && contains_all(Z, points),
                  b + ": " + method + " contains the cloud, n = " + std::to_string(n));
        }
    }
    const Tensor points = cloud(rng, 3, 40);
    check(contains_all(Zonotope::enclose_points(points), points), b + ": the default is maiga");
    const Zonotope maiga = Zonotope::enclose_points(cloud(rng, 2, 30), "maiga");
    check(maiga.generators().shape().back() >= 2,
          b + ": maiga: at least the n generators of the box");
}

/// A method that does not exist, an empty cloud and a frame of the wrong size are refused; with
/// the checks off the result is the same.
void checks_and_their_switch() {
    const Tensor points({{0.0, 1.0}, {0.0, 1.0}});
    check(test::raises([&] { Zonotope::enclose_points(points, "other"); }, Error::wrong_value),
          "enclose_points: method");
    check(
        test::raises([&] { Zonotope::enclose_points(Tensor::zeros({2, 0})); }, Error::wrong_value),
        "enclose_points: no points");
    check(test::raises([&] { Zonotope::enclose_points(points, Tensor::eye(3)); },
                       Error::dimension_mismatch),
          "enclose_points: frame");
    const Tensor on = Zonotope::enclose_points(points, "stursberg").generators();
    ChecksScope off(false);
    check(close(Zonotope::enclose_points(points, "stursberg").generators(), on, 0),
          "enclose_points: checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        enclose_points_known(b);
        enclose_points_contain_the_cloud(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope enclose_points");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
