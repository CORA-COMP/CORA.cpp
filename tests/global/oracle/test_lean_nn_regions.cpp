// test_lean_nn_regions - the polynomials of a ReLU region by region against CORALean
//
// ORACLE: nn/layers/nonlinear/@nnActivationLayer/findRegionPolys.m
//
// CORA.cpp against the oracle, bit for bit: find_region_polys of a ReLU on random tolerances,
// orders and domains and on cases that lower the order to a line and to a constant. Every region
// between the outer two (its ends, the regression polynomial and its radius) is compared, and a
// domain the halving cannot close fails on both sides (CORA's notConverged). The fit takes only
// arithmetic and sqrt (the Vandermonde powers are repeated products), so no libm is compared.
//
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/neural_network.h"
#include "cora/nn/neural_network/neural_network.h"
#include "oracle_testing.h"

#include <cmath>
#include <exception>
#include <limits>
#include <string>
#include <vector>

using namespace cora;

namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A case: the tolerance, the order and the domain [l_max, u_max].
struct Case {
    double tol;
    int order;
    double l_max, u_max;
};

/// CORA.cpp's regions of the case against the oracle's, or both failing.
void aux_check(const Case &c) {
    const std::string what = "findRegionPolys tol " + std::to_string(c.tol) + " order " +
                             std::to_string(c.order) + " [" + std::to_string(c.l_max) + ", " +
                             std::to_string(c.u_max) + "]";
    const double inf = std::numeric_limits<double>::infinity();
    // the outer regions are the arguments; the oracle replies the ones between
    std::vector<RegionPoly> mine;
    const bool failed = test::raises(
        [&] {
            mine = NeuralNetwork::find_region_polys(Activation::relu, c.tol, c.order, c.l_max,
                                                    c.u_max, {0.0}, 0.0, {1.0, 0.0}, 0.0);
        },
        Error::not_converged);
    std::vector<RegionPoly> theirs;
    bool refused = false;
    try {
        theirs = lean::find_region_polys(c.tol, c.order, c.l_max, c.u_max);
    } catch (const std::exception &) {
        refused = true;
    }
    test::check(failed == refused, what + ": both converge or neither");
    if (failed || refused) return;
    // every region between the outer two, in order
    test::check(mine.size() == theirs.size() + 2 && mine.front().l == -inf && mine.back().u == inf,
                what + ": the regions and the outer two");
    for (std::size_t i = 0; i < theirs.size() && i + 1 < mine.size(); ++i) {
        const RegionPoly &a = mine[i + 1], &b = theirs[i];
        const std::string r = what + " region " + std::to_string(i);
        oracle::check(r + " l", a.l, b.l);
        oracle::check(r + " u", a.u, b.u);
        oracle::check(r + " d", a.d, b.d);
        test::check(a.p.size() == b.p.size(), r + ": the order");
        for (std::size_t j = 0; j < a.p.size() && j < b.p.size(); ++j)
            oracle::check(r + " p" + std::to_string(j), a.p[j], b.p[j]);
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    oracle::run("nnLayer.findRegionPolys", [&](oracle::Random &rnd) {
        // a cubic next to the zero piece, a constant, a line, exact pieces and no convergence
        for (const Case &c :
             {Case{0.125, 3, -3.0, 2.5}, Case{0.25, 4, -0.125, 0.375}, Case{0.5, 2, -1.0, 2.0},
              Case{0.125, 1, -3.0, 1.0}, Case{0.0625, 2, -1.0, 2.0}})
            aux_check(c);
        // random tolerances, orders and domains around 0
        for (int t = 0; t < 40; ++t) {
            const double tol = rnd.uniform(0.05, 1.0);
            const int order = 1 + int(std::floor(rnd.uniform(0.0, 4.0)));
            const double l = -rnd.uniform(0.1, 4.0), u = rnd.uniform(0.1, 4.0);
            aux_check({tol, order, l, u});
        }
    });
    return test::finish("lean nn regions");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
