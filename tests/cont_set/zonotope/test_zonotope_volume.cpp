// test_zonotope_volume - zonotope volume: the volume of a zonotope
//
// CORA: contSet/@zonotope/volume_.m
// PROPERTY: contSet/@zonotope/volume_.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// CORA's unit test: Z = zonotope([-4 -3 -2 -1; 1 2 3 4]) has volume 80, the Alamo value 48.98...
void cora_unit_test(const std::string &b) {
    const Zonotope Z(column({-4.0, 1.0}), Tensor({{-3.0, -2.0, -1.0}, {2.0, 3.0, 4.0}}));
    check(close(Z.volume(), std::vector<double>{80}, 1e-12), b + ": exact 80");
    check(close(Z.volume("alamo"), std::vector<double>{48.9897948556635612}, 1e-12),
          b + ": alamo 4 sqrt(150)");
    // the box of Z is larger
    check(Z.volume().item() < 8 * 12, b + ": less than the box");
}

/// A cube has volume 2^n; fewer generators than dimensions and a flat set have volume 0.
void by_hand(const std::string &b) {
    const Zonotope cube(column({1.0, 2.0, 3.0}),
                        Tensor({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}));
    check(close(cube.volume(), std::vector<double>{8}, 1e-14), b + ": the cube, exact");
    check(close(cube.volume("alamo"), std::vector<double>{8}, 1e-14), b + ": the cube, alamo");
    const Zonotope few(column({0.0, 0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}}));
    check(close(few.volume(), std::vector<double>{0}, 0), b + ": fewer generators than dimensions");
    const Zonotope flat(column({0.0, 0.0}), Tensor({{1.0, 2.0, 3.0}, {2.0, 4.0, 6.0}}));
    check(close(flat.volume(), std::vector<double>{0}, 0), b + ": exact: rank 1 in the plane");
    check(close(flat.volume("alamo"), std::vector<double>{0}, 0), b + ": alamo: rank 1");
    const Zonotope point(column({1.0, 2.0}), Tensor::zeros({2, 0}));
    check(close(point.volume(), std::vector<double>{0}, 0), b + ": a point");
}

/// Stated property: in the plane the volume is the area of the polygon of vertices(), in three
/// dimensions with three generators it is 8 |det G|.
void the_volume_is_the_area(const std::string &b) {
    cora::Rng rng(61);
    for (int m = 2; m <= 6; ++m) {
        const Zonotope Z = Zonotope::generate_random(rng, 2, std::nullopt, m);
        const Polygon p = Z.vertices()[0];
        double area = 0;
        for (std::size_t i = 0; i < p.size(); ++i) {
            const Point &u = p[i], &v = p[(i + 1) % p.size()];
            area += u[0] * v[1] - v[0] * u[1];
        }
        check(close(Z.volume().item(), std::abs(area) / 2, 1e-9),
              b + ": the area, m = " + std::to_string(m));
        check(Z.volume("alamo").item() <= Z.volume().item() + 1e-9, b + ": alamo is a lower bound");
    }
    const Zonotope P = Zonotope::generate_random(rng, 3, std::nullopt, 3);
    const std::vector<double> g = P.generators().data();
    const double det = g[0] * (g[4] * g[8] - g[5] * g[7]) - g[1] * (g[3] * g[8] - g[5] * g[6]) +
                       g[2] * (g[3] * g[7] - g[4] * g[6]);
    check(close(P.volume().item(), 8 * std::abs(det), 1e-9), b + ": three generators: 8 |det G|");
}

/// A batch is the volume of each member, including the members of rank below n.
void a_batch(const std::string &b) {
    cora::Rng rng(62);
    const Zonotope A = Zonotope::generate_random(rng, 2, std::nullopt, 4);
    const Zonotope flat(column({0.0, 0.0}), Tensor({{1.0, 2.0, 3.0, 1.0}, {2.0, 4.0, 6.0, 2.0}}));
    const Zonotope batch = Zonotope::stack({A, flat});
    for (const std::string method : {"exact", "alamo"}) {
        const Tensor v = batch.volume(method);
        check(v.shape() == Shape{2, 1, 1}, b + ": batch shape");
        check(close(v.slice(0, 0, 1).reshape({1, 1}), A.volume(method), 1e-12),
              b + ": member 1 (" + method + ")");
        check(close(v.slice(0, 1, 2).reshape({1, 1}), std::vector<double>{0}, 0),
              b + ": the flat member has volume 0 (" + method + ")");
    }
}

/// A wrong method or order is refused, 'reduce' is not supported; with the checks off the same
/// call gives the same result.
void checks_and_their_switch() {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0, 0.0, 1.0}, {0.0, 1.0, 1.0}}));
    check(test::raises([&] { Z.volume("fast"); }, Error::wrong_value), "volume: unknown method");
    check(test::raises([&] { Z.volume("exact", 0); }, Error::wrong_value), "volume: order 0");
    check(test::raises([&] { Z.volume("reduce", 1); }, Error::not_supported), "volume: reduce");
    const Tensor on = Z.volume();
    ChecksScope off(false);
    check(close(Z.volume(), on, 0), "volume: same with checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        cora_unit_test(b);
        by_hand(b);
        the_volume_is_the_area(b);
        a_batch(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope volume");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
