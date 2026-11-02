// test_zonotope_vertices - zonotope vertices: the corners of a planar zonotope, counter-clockwise
//
// CORA: contSet/@zonotope/vertices_.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;

namespace {

double area(const Polygon &p) {
    double a = 0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        const Point &u = p[i], &v = p[(i + 1) % p.size()];
        a += u[0] * v[1] - v[0] * u[1];
    }
    return a / 2;
}

/// The corners attain the support function in every direction, and run counter-clockwise.
void vertices_attain_the_support(const std::string &b) {
    cora::Rng rng(8);
    const Zonotope Z = Zonotope::generate_random(2, 4, rng);
    const std::vector<Polygon> V = Z.vertices();
    check(V.size() == 1, b + ": one polygon for one zonotope");
    check(V[0].size() == 8, b + ": two vertices per generator direction");
    check(area(V[0]) > 0, b + ": counter-clockwise");
    for (int k = 0; k < 20; ++k) {
        const std::vector<double> d = test::random_direction(rng, 2);
        double best = -1e300;
        for (const Point &p : V[0]) best = std::max(best, d[0] * p[0] + d[1] * p[1]);
        check(close(best, test::support(Z, d), 1e-9), b + ": corners reach the support function");
    }
}

/// Without generators the zonotope is a point; parallel generators add no corners.
void vertices_of_degenerate_zonotopes(const std::string &b) {
    const Zonotope point(test::column({1, 2}), Tensor::zeros({2, 1}));
    check(point.vertices()[0].size() == 1, b + ": a point has one vertex");
    const Zonotope segment(test::column({0, 0}), Tensor({{1, 2}, {0, 0}}));
    check(segment.vertices()[0].size() == 2, b + ": parallel generators make a segment");
}

void vertices_need_two_dimensions(const std::string &b) {
    cora::Rng rng(1);
    check(test::throws([&] { Zonotope::generate_random(3, 2, rng).vertices(); }),
          b + ": a three-dimensional zonotope has no polygon");
}

/// A set of three dimensions is refused; with the checks off the polygon is the same.
void checks_and_their_switch() {
    const Zonotope S3 =
        Zonotope(Tensor({1.0, 0.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}, {0.0, 0.0}}));
    check(test::raises([&] { S3.vertices(); }, Error::not_supported), "vertices: three dimensions");
    const Zonotope S = Zonotope(Tensor({1.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}}));
    const std::size_t on = S.vertices()[0].size();
    ChecksScope off(false);
    check(S.vertices()[0].size() == on, "vertices: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        vertices_attain_the_support(b);
        vertices_of_degenerate_zonotopes(b);
        vertices_need_two_dimensions(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope vertices");
}
