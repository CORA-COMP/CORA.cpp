// test_lean_zonotope_maps_sound - radius, rotate and quad_map of Zonotope against the oracle,
// within a stated bound
//
// CORA: contSet/@zonotope/radius.m
// CORA: contSet/@zonotope/rotate.m
// CORA: contSet/@zonotope/quadMap.m
//
// These cannot match the oracle bit for bit, by construction, so there is no ORACLE tag: radius
// takes square roots, which the oracle computes by Newton steps rounded up (a few ulp above the
// correctly rounded ones); rotate takes cos and sin of the platform's math library, where the
// oracle encloses them (an interval rotation matrix); quad_map is CORA's zonotope map, where
// CORALean maps through a polynomial zonotope (a different, wider set), so it only has to lie
// inside the oracle's enclosure. Radius and rotate are compared within a relative 1e-12.

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/oracle/zonotope.h"
#include "oracle_testing.h"

#include <cmath>

using namespace cora;
namespace oracle = test::oracle;

namespace {

/// The slack of the comparisons: a relative 1e-12 (a few thousand ulp), for the Newton square
/// roots, cos and sin, and the rounding of sums of up to 50 products.
constexpr double kSlack = 1e-12;

/// The interval hull of mine lies in the oracle's enclosure up to the slack, and is not tighter
/// than a relative 1e-9 of the scale from it.
void aux_hull_agrees(const std::string &what, const Zonotope &mine, const lean::Zonotope &theirs) {
    const Interval a = mine.interval();
    const auto [inf, sup] = theirs.enclosure().interval().gather();
    const std::vector<double> lo = a.inf().data(), hi = a.sup().data();
    const std::vector<double> blo = inf.data(), bhi = sup.data();
    for (std::size_t i = 0; i < lo.size(); ++i) {
        const double scale = 1 + std::abs(bhi[i]) + std::abs(blo[i]);
        test::check(blo[i] <= lo[i] + kSlack * scale && hi[i] <= bhi[i] + kSlack * scale,
                    what + ": outside the enclosure");
        test::check(lo[i] - blo[i] <= 1e-9 * scale && bhi[i] - hi[i] <= 1e-9 * scale,
                    what + ": far inside the enclosure");
    }
}

} // namespace

int main() {
    if (!oracle::available()) return 0;
    const std::vector<std::pair<int, int>> sizes{{2, 3}, {5, 30}, {20, 50}};

    oracle::run("zonotope.radius", [&](oracle::Random &rnd) {
        for (const auto [n, m] : sizes) {
            const Zonotope Z = rnd.zonotope(n, m);
            const double mine = Z.radius().item(), bound = lean::Zonotope(Z).radius();
            test::check(mine <= bound * (1 + kSlack), "radius: " + std::to_string(mine) +
                                                          " exceeds the bound " +
                                                          std::to_string(bound));
            test::check(mine >= bound * (1 - 1e-6), "radius: the bound " + std::to_string(bound) +
                                                        " is far above " + std::to_string(mine));
        }
    });

    oracle::run("zonotope.quadMap", [&](oracle::Random &rnd) {
        for (const auto [n, m] : {std::pair{2, 3}, {3, 4}, {4, 6}}) {
            const Zonotope Z = rnd.zonotope(n, m);
            const std::vector<Tensor> Q{rnd.matrix(n, n), rnd.matrix(n, n)};
            std::vector<lean::Tensor> LQ;
            LQ.reserve(Q.size());
            for (const Tensor &M : Q) LQ.push_back(lean::Tensor::from(M));
            oracle::check_enclosed("zonotope.quadMap", Z.quad_map(Q),
                                   lean::Zonotope(Z).quad_map(LQ));
        }
    });

    oracle::run("zonotope.rotate", [&](oracle::Random &rnd) {
        for (const auto [n, m] : sizes) {
            const Zonotope Z = rnd.zonotope(n, m);
            const double angle = rnd.uniform(-3, 3);
            aux_hull_agrees("rotate", Z.rotate({0, int64_t(n - 1)}, angle),
                            lean::Zonotope(Z).rotate(20, 0, n - 1, angle));
        }
    });
    return test::finish("lean zonotope maps (bounds)");
}
