// test_zonotope_vertices_3d - zonotope vertices_3d: the corners of a three-dimensional zonotope

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::close;

namespace {

/// The corners attain the support function in every direction, and number m (m - 1) + 2 for
/// generic generators.
void corners_attain_the_support(const std::string &b) {
    cora::Rng rng(8);
    for (int m : {3, 4, 7}) {
        const Zonotope Z = Zonotope::generate_random(3, m, rng);
        const std::vector<Point3> V = Z.vertices_3d()[0];
        check(V.size() == static_cast<std::size_t>(m * (m - 1) + 2),
              b + ": m (m - 1) + 2 corners for m = " + std::to_string(m));
        for (int k = 0; k < 20; ++k) {
            const std::vector<double> d = test::random_direction(rng, 3);
            double best = -1e300;
            for (const Point3 &p : V)
                best = std::max(best, d[0] * p[0] + d[1] * p[1] + d[2] * p[2]);
            check(close(best, test::support(Z, d), 1e-9),
                  b + ": corners reach the support function");
        }
    }
}

/// A cube with a diagonal bar has twelve corners, and a flat or thin zonotope gives its polygon,
/// segment or point.
void degenerate_zonotopes(const std::string &b) {
    const Zonotope cube(test::column({0, 0, 0}),
                        Tensor({{1, 0, 0, 1}, {0, 1, 0, 1}, {0, 0, 1, 0}}));
    check(cube.vertices_3d()[0].size() == 12, b + ": a cube with a diagonal bar has 12 corners");
    const Zonotope flat(test::column({0, 0, 0}), Tensor({{1, 0}, {0, 1}, {0, 0}}));
    check(flat.vertices_3d()[0].size() == 4, b + ": two generators make a square");
    const Zonotope segment(test::column({0, 0, 0}), Tensor({{1, 2}, {1, 2}, {0, 0}}));
    check(segment.vertices_3d()[0].size() == 2, b + ": parallel generators make a segment");
    const Zonotope point(test::column({1, 2, 3}), Tensor::zeros({3, 1}));
    check(point.vertices_3d()[0].size() == 1, b + ": no generator, one point");
}

/// A zonotope of other dimensions is refused; with the checks off the corners are the same.
void checks_and_their_switch() {
    const Zonotope S2 = Zonotope(Tensor({1.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}}));
    check(test::raises([&] { S2.vertices_3d(); }, Error::not_supported),
          "vertices_3d: two dimensions");
    const Zonotope S =
        Zonotope(Tensor({1.0, 0.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}, {0.0, 0.0}}));
    const std::size_t on = S.vertices_3d()[0].size();
    ChecksScope off(false);
    check(S.vertices_3d()[0].size() == on, "vertices_3d: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        corners_attain_the_support(b);
        degenerate_zonotopes(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope vertices_3d");
}
