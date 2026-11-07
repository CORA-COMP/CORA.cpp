// test_lean_con_zonotope_enclosure - the operations of ConZonotope that the CORALean oracle builds
//
// in another form, inside its interval hull
//
// convHull as a hull with inequality rows on the multipliers, enclose with inequality rows,
// intersectStrip with the gain's product rounded in another order, interval as a bound from a
// program, reduce and quadMap with CORALean's layout of the new factors, split with bounds from a
// program: the center c matches bit for bit where kept and CORA.cpp's box lies inside the
// oracle's. cubMap sums all N^3 terms in closed form: its hull lies inside the oracle's up to a
// relative 1e-10. None is bit-exact, so there is no ORACLE tag.
//
// CORA: contSet/@conZonotope/convHull_.m
// CORA: contSet/@conZonotope/enclose.m
// CORA: contSet/@conZonotope/intersectStrip.m
// CORA: contSet/@conZonotope/cubMap.m
// CORA: contSet/@conZonotope/interval.m
// CORA: contSet/@conZonotope/reduce.m
// CORA: contSet/@conZonotope/quadMap.m
// CORA: contSet/@conZonotope/split.m
// CORA authors: Dmitry Grebenyuk, Matthias Althoff, Mark Wetzlinger, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   07-November-2026
// Reviewed:      ---

#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "cora/global/oracle/con_zonotope.h"
#include "oracle_testing.h"

#include <cmath>

using namespace cora;


namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The result of an operation that builds another form of the set: its center agrees and its box
/// lies in the oracle's box.
void aux_check_hull(const ConZonotope &mine, const lean::ConZonotope &theirs) {
    oracle::check(oracle::op_name() + ".c", mine.c(), theirs.nominal().c());
    oracle::check_inside(mine.interval(), theirs.interval());
}

/// The hull of CORA.cpp's cubic map inside the oracle's up to a relative 1e-10: CORA lists the
/// products x_i x_j x_k once and sums them in its order, CORALean sums N^3 terms in closed form, so
/// the bounds differ by a few hundred ulp of the sums of cubes.
void aux_check_cub_map(const ConZonotope &mine, const lean::ConZonotope &theirs) {
    const Interval hull = mine.interval();
    const auto [inf, sup] = theirs.interval().gather();
    const std::vector<double> a = hull.inf().data(), b = inf.data(), c = hull.sup().data(),
                              d = sup.data();
    constexpr double slack = 1e-10;
    bool ok = a.size() == b.size();
    for (std::size_t i = 0; ok && i < a.size(); ++i)
        ok = a[i] >= b[i] - slack * (1 + std::abs(b[i])) &&
             c[i] <= d[i] + slack * (1 + std::abs(d[i]));
    test::check(ok, oracle::op_name() + ": the hull inside the oracle's up to a relative 1e-10");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    oracle::run("conZonotope.convHull", [](oracle::Random &rnd) {
        const ConZonotope a = rnd.con_zonotope(3, 5, 2), b = rnd.con_zonotope(3, 4, 1);
        aux_check_hull(a.convex_hull(b), lean::ConZonotope(a).convex_hull(b));
    });
    oracle::run("conZonotope.enclose", [](oracle::Random &rnd) {
        const ConZonotope a = rnd.con_zonotope(3, 5, 2), b = rnd.con_zonotope(3, 3, 0);
        const Tensor M = rnd.matrix(3, 3);
        aux_check_hull(a.enclose(M, b), lean::ConZonotope(a).enclose(M, b));
    });
    // strip intersection
    oracle::run("conZonotope.intersectStrip", [](oracle::Random &rnd) {
        const ConZonotope a = rnd.con_zonotope(3, 5, 2);
        const Tensor C = rnd.matrix(2, 3), y = rnd.vector(2), L = rnd.matrix(3, 2);
        const Tensor phi = rnd.matrix(2, 1).abs();
        aux_check_hull(a.intersect_strip(C, phi, y, L),
                       lean::ConZonotope(a).intersect_strip(C, phi, y, L));
    });
    oracle::run("conZonotope.cubMap", [](oracle::Random &rnd) {
        const ConZonotope a = rnd.con_zonotope(2, 3, 0);
        std::vector<std::vector<Tensor>> T(2);
        std::vector<std::vector<lean::Tensor>> theirs_T(2);
        for (int i = 0; i < 2; ++i)
            for (int k = 0; k < 2; ++k) {
                T[i].push_back(rnd.matrix(2, 2));
                theirs_T[i].emplace_back(T[i].back());
            }
        aux_check_cub_map(a.cubic_map(T), lean::ConZonotope(a).cub_map(theirs_T));
    });
    oracle::run("conZonotope.interval", [](oracle::Random &rnd) {
        const ConZonotope a = rnd.con_zonotope(3, 5, 2);
        // interval hull and bounds
        oracle::check_inside(a.interval(), lean::ConZonotope(a).interval());
    });
    // reduce, quadMap and split build another form of the set: CORALean's layout of the new factors
    oracle::run("conZonotope.reduce", [](oracle::Random &rnd) {
        const ConZonotope a = rnd.con_zonotope(2, 8, 2);
        oracle::check_inside(a.reduce(2).interval(), lean::ConZonotope(a).reduce(2).interval());
    });
    oracle::run("conZonotope.quadMap", [](oracle::Random &rnd) {
        const ConZonotope a = rnd.con_zonotope(2, 3, 0);
        const std::vector<Tensor> Q{rnd.matrix(2, 2), rnd.matrix(2, 2)};
        const std::vector<lean::Tensor> theirs_Q(Q.begin(), Q.end());
        oracle::check_inside(a.quad_map(Q).interval(),
                             lean::ConZonotope(a).quad_map(theirs_Q).interval());
    });
    oracle::run("conZonotope.split", [](oracle::Random &rnd) {
        const ConZonotope a = rnd.con_zonotope(3, 5, 2);
        const auto mine = a.split(1);
        const auto theirs = lean::ConZonotope(a).split(1);
        oracle::check_inside(mine.first.interval(), theirs.first.interval());
        oracle::check_inside(mine.second.interval(), theirs.second.interval());
    });
    return test::finish("lean con zonotope enclosure");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
