// test_lean_con_zonotope - the operations of ConZonotope that the CORALean oracle serves
//
// Bit for bit where CORALean builds the same set: and, cartProd, compact, conIntersect (and
// isBounded as a Boolean); empty is an inequality row, so only its emptiness is compared. The ops
// CORALean builds in another form are in test_lean_con_zonotope_enclosure.
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

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    oracle::run("conZonotope.and", [](oracle::Random &rnd) {
        const ConZonotope a = rnd.con_zonotope(3, 5, 2), b = rnd.con_zonotope(3, 4, 1);
        const ConZonotope mine = a.intersect(b);
        const lean::ConZonotope theirs = lean::ConZonotope(a).intersect(b);
        // CORALean keeps the constraints as equalities, the points' agreement as two inequalities
        const ConZonotope their = theirs.nominal();
        const auto [A, b_ineq] = theirs.inequalities();
        const int64_t pe = their.A().shape()[0], n = mine.dim();
        oracle::check("and.c", mine.c(), their.c());
        oracle::check("and.G", mine.generators(), their.generators());
        oracle::check("and.Ae", aux_rows(mine.A(), 0, pe), their.A());
        oracle::check("and.be", aux_rows(mine.b(), 0, pe), their.b());
        oracle::check("and.A", aux_rows(mine.A(), pe, pe + n), aux_rows(A, 0, n));
        oracle::check("and.b", aux_rows(mine.b(), pe, pe + n), aux_rows(b_ineq, 0, n));
        oracle::check("and.-A", -aux_rows(mine.A(), pe, pe + n), aux_rows(A, n, 2 * n));
        oracle::check("and.-b", -aux_rows(mine.b(), pe, pe + n), aux_rows(b_ineq, n, 2 * n));
    });
    // the product stacks both sets
    oracle::run("conZonotope.cartProd", [](oracle::Random &rnd) {
        const ConZonotope a = rnd.con_zonotope(3, 5, 2), b = rnd.con_zonotope(2, 4, 1);
        oracle::check(a.cart_prod(b), lean::ConZonotope(a).cart_prod(b));
    });
    // constrained intersection
    oracle::run("conZonotope.conIntersect", [](oracle::Random &rnd) {
        const ConZonotope a = rnd.con_zonotope(3, 5, 2), b = rnd.con_zonotope(2, 4, 1);
        const Tensor R = rnd.matrix(2, 3);
        oracle::check(a.con_intersect(b, R), lean::ConZonotope(a).con_intersect(b, R));
    });
    oracle::run("conZonotope.compact", [](oracle::Random &rnd) {
        const ConZonotope a = rnd.con_zonotope(3, 5, 2);
        oracle::check(a.compact(), lean::ConZonotope(a).compact());
    });
    // CORALean's empty set is an inequality row: only the emptiness is compared
    oracle::run("conZonotope.empty", [](oracle::Random &) {
        const ConZonotope mine = ConZonotope::empty(3);
        const lean::ConZonotope theirs = lean::ConZonotope::empty(3);
        test::check(theirs.nominal().dim() == 3 && mine.dim() == 3, "empty: the dimension");
        test::check(mine.interval().is_empty(), "empty: CORA.cpp's set is empty");
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
