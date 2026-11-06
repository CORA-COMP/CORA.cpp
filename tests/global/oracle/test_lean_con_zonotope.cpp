// test_lean_con_zonotope - the operations of ConZonotope that the CORALean oracle serves
//
// Bit for bit where CORALean builds the same set: and, cartProd, compact, conIntersect (and
// isBounded as a Boolean). The others it builds in another form, so they are checked as enclosures
// of CORA.cpp's set (the interval hulls; no ORACLE tag): convHull as a hull with inequality rows
// on the multipliers, enclose with inequality rows, cubMap with all N^3 terms and the sums in
// closed form, intersectStrip with the gain's product rounded in another order, empty as an
// inequality row, interval as an outward-rounded bound from a program; the center c matches.
//
// ORACLE: contSet/@conZonotope/and_.m
// ORACLE: contSet/@conZonotope/cartProd_.m
// ORACLE: contSet/@conZonotope/compact_.m
// ORACLE: contSet/@conZonotope/conIntersect.m
// ORACLE: contSet/@conZonotope/isBounded.m
// ORACLE: contSet/@zonotope/conZonotope.m
// ORACLE: contSet/@interval/conZonotope.m
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
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

/// The rows [from, to) of a matrix.
Tensor aux_rows(const Tensor &T, int64_t from, int64_t to) { return T.slice(0, from, to); }

/// The result of an operation that builds another form of the set: in the exact mode its center
/// agrees bit for bit and the set's hull lies in the oracle's hull; in the sound mode the hull
/// alone.
void aux_check_hull(const ConZonotope &mine, const lean::ConZonotope &theirs) {
    if (oracle::mode() == oracle::Mode::exact)
        test::check(oracle::same_bits(oracle::op_name() + ".c", mine.c(), theirs.nominal().c()),
                    oracle::op_name() + ": the same center");
    oracle::check_enclosed(oracle::op_name(), mine, theirs);
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
    oracle::run("conZonotope.and", [](oracle::Random &rnd) {
        const ConZonotope a = rnd.con_zonotope(3, 5, 2), b = rnd.con_zonotope(3, 4, 1);
        const ConZonotope mine = a.intersect(b);
        const lean::ConZonotope theirs = lean::ConZonotope(a).intersect(b);
        if (oracle::mode() == oracle::Mode::sound)
            return oracle::check_enclosed("and", mine, theirs);
        // CORALean keeps the constraints of both sets as equalities, the agreement of the points
        // as the two inequalities c1 + G1 x1 - c2 - G2 x2 <= 0 and >= 0, CORA as equalities
        const ConZonotope their = theirs.nominal();
        const auto [A, b_ineq] = theirs.inequalities();
        const int64_t pe = their.A().shape()[0], n = mine.dim();
        oracle::check_exact("and.c", mine.c(), their.c());
        oracle::check_exact("and.G", mine.generators(), their.generators());
        oracle::check_exact("and.Ae", aux_rows(mine.A(), 0, pe), their.A());
        oracle::check_exact("and.be", aux_rows(mine.b(), 0, pe), their.b());
        oracle::check_exact("and.A", aux_rows(mine.A(), pe, pe + n), aux_rows(A, 0, n));
        oracle::check_exact("and.b", aux_rows(mine.b(), pe, pe + n), aux_rows(b_ineq, 0, n));
        oracle::check_exact("and.-A", -aux_rows(mine.A(), pe, pe + n), aux_rows(A, n, 2 * n));
        oracle::check_exact("and.-b", -aux_rows(mine.b(), pe, pe + n), aux_rows(b_ineq, n, 2 * n));
    });
    oracle::run("conZonotope.cartProd", [](oracle::Random &rnd) {
        const ConZonotope a = rnd.con_zonotope(3, 5, 2), b = rnd.con_zonotope(2, 4, 1);
        oracle::check(a.cart_prod(b), lean::ConZonotope(a).cart_prod(b));
    });
    oracle::run("conZonotope.convHull", [](oracle::Random &rnd) {
        const ConZonotope a = rnd.con_zonotope(3, 5, 2), b = rnd.con_zonotope(3, 4, 1);
        aux_check_hull(a.convex_hull(b), lean::ConZonotope(a).convex_hull(b));
    });
    oracle::run("conZonotope.conIntersect", [](oracle::Random &rnd) {
        const ConZonotope a = rnd.con_zonotope(3, 5, 2), b = rnd.con_zonotope(2, 4, 1);
        const Tensor R = rnd.matrix(2, 3);
        oracle::check(a.con_intersect(b, R), lean::ConZonotope(a).con_intersect(b, R));
    });
    oracle::run("conZonotope.compact", [](oracle::Random &rnd) {
        const ConZonotope a = rnd.con_zonotope(3, 5, 2);
        oracle::check(a.compact(), lean::ConZonotope(a).compact());
    });
    oracle::run("conZonotope.empty", [](oracle::Random &) {
        const ConZonotope mine = ConZonotope::empty(3);
        const lean::ConZonotope theirs = lean::ConZonotope::empty(3);
        test::check(theirs.nominal().dim() == 3 && mine.dim() == 3, "empty: the dimension");
        test::check(mine.interval().is_empty(), "empty: CORA.cpp's set is empty");
    });
    oracle::run("conZonotope.enclose", [](oracle::Random &rnd) {
        const ConZonotope a = rnd.con_zonotope(3, 5, 2), b = rnd.con_zonotope(3, 3, 0);
        const Tensor M = rnd.matrix(3, 3);
        aux_check_hull(a.enclose(M, b), lean::ConZonotope(a).enclose(M, b));
    });
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
        oracle::check_enclosed("interval", a.interval(), lean::ConZonotope(a).interval());
    });
    oracle::run("conZonotope.isBounded", [](oracle::Random &rnd) {
        const ConZonotope a = rnd.con_zonotope(3, 5, 2);
        oracle::check(a.is_bounded(), lean::ConZonotope(a).is_bounded());
    });
    oracle::run("zonotope.conZonotope", [](oracle::Random &rnd) {
        const Zonotope Z = rnd.zonotope(3, 4);
        oracle::check(ConZonotope(Z), lean::ConZonotope::from_zonotope(lean::Zonotope(Z)));
    });
    oracle::run("interval.conZonotope", [](oracle::Random &rnd) {
        const Interval I = rnd.box(3);
        oracle::check(ConZonotope(I), lean::ConZonotope::from_interval(lean::Interval(I)));
    });
    return test::finish("lean con zonotope");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
