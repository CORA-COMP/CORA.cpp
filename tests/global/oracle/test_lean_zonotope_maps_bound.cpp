// test_lean_zonotope_maps_bound - radius and quad_map of Zonotope against the oracle,
//
// within a stated bound
//
// CORA: contSet/@zonotope/radius.m
// CORA: contSet/@zonotope/quadMap.m
//
// These cannot match the oracle bit for bit, by construction, so there is no ORACLE tag: radius
// takes square roots, which the oracle computes by Newton steps rounded up (a few ulp above the
// correctly rounded ones); quad_map is CORA's zonotope map, where
// CORALean maps through a polynomial zonotope (a different, wider set), so its box only has to
// lie inside the oracle's. Radius is compared within a relative 1e-12.
//
// CORA authors: Matthias Althoff, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   21-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/oracle/zonotope.h"
#include "oracle_testing.h"

#include <cmath>

using namespace cora;


namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The slack of the comparisons: a relative 1e-12 (a few thousand ulp), for the Newton square
/// roots and the rounding of sums of up to 50 products.
constexpr double kSlack = 1e-12;

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    const std::vector<std::pair<int, int>> sizes{{2, 3}, {5, 30}, {20, 50}};

    oracle::run("zonotope.radius", [&](oracle::Random &rnd) {
        for (const auto &[n, m] : sizes) {
            const Zonotope Z = rnd.zonotope(n, m);
            const double mine = Z.ball_radius().item(), bound = lean::Zonotope(Z).radius();
            test::check(mine <= bound * (1 + kSlack), "radius: " + std::to_string(mine) +
                                                          " exceeds the bound " +
                                                          std::to_string(bound));
            test::check(mine >= bound * (1 - 1e-6), "radius: the bound " + std::to_string(bound) +
                                                        " is far above " + std::to_string(mine));
        }
    });

    oracle::run("zonotope.quadMap", [&](oracle::Random &rnd) {
        for (const auto &[n, m] : {std::pair{2, 3}, {3, 4}, {4, 6}}) {
            const Zonotope Z = rnd.zonotope(n, m);
            const std::vector<Tensor> Q{rnd.matrix(n, n), rnd.matrix(n, n)};
            std::vector<lean::Tensor> LQ;
            LQ.reserve(Q.size());
            // the quadratic map of zonotopes
            for (const Tensor &M : Q) LQ.push_back(lean::Tensor::from(M));
            oracle::check_inside(Z.quad_map(Q).interval(),
                                 lean::Zonotope(Z).quad_map(LQ).interval());
        }
    });

    return test::finish("lean zonotope maps (bounds)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
