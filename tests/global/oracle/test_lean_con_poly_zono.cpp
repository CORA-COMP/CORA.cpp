// test_lean_con_poly_zono - ConPolyZono's conversions, maps and set operations
//
// against the CORALean oracle, bit for bit
//
// Every operation follows CORALean's float definition of ConPolyZono (conPolyZono.* ops): the
// conversions of a zonotope, a polynomial zonotope and a constrained zonotope, the map, the sums,
// the product, the intersection (the agreement rows), the linear combination and the hull, the
// projections, compact, Girard's reduction, the box, the zonotope and the polynomial part.
//
// ORACLE: contSet/@conPolyZono/conPolyZono.m
// ORACLE: contSet/@conPolyZono/origin.m
// ORACLE: contSet/@conPolyZono/empty.m
// ORACLE: contSet/@conPolyZono/dim.m
// ORACLE: contSet/@conPolyZono/center.m
// ORACLE: contSet/@conPolyZono/mtimes.m
// ORACLE: contSet/@conPolyZono/plus.m
// ORACLE: contSet/@conPolyZono/exactPlus.m
// ORACLE: contSet/@conPolyZono/cartProd_.m
// ORACLE: contSet/@conPolyZono/and_.m
// ORACLE: contSet/@conPolyZono/linComb.m
// ORACLE: contSet/@conPolyZono/convHull_.m
// ORACLE: contSet/@conPolyZono/project.m
// ORACLE: contSet/@conPolyZono/lift_.m
// ORACLE: contSet/@conPolyZono/compact_.m
// ORACLE: contSet/@conPolyZono/reduce.m
// ORACLE: contSet/@conPolyZono/interval.m
// ORACLE: contSet/@conPolyZono/zonotope.m
// ORACLE: contSet/@conPolyZono/polyZonotope.m
// ORACLE: contSet/@conPolyZono/isequal.m
// ORACLE: contSet/@conPolyZono/representsa_.m
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_poly_zono_oracle_testing.h"
#include "cora/global/oracle/con_poly_zono.h"

using namespace cora;
namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The two random sets of one test: with and without independent generators.
ConPolyZono aux_set(oracle::Random &rnd, int64_t q = 2) {
    return oracle::con_poly_zono(rnd, {3, 4, 3, q, 2, 3});
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    // the conversions, and the two sets of no generator
    oracle::run("conPolyZono.ofZonotope", [](oracle::Random &rnd) {
        const Zonotope Z = rnd.zonotope(3, 4);
        oracle::check("zonotope", ConPolyZono(Z), lean::ConPolyZono::from_zonotope(Z));
        const PolyZonotope P(rnd.point(3), rnd.matrix(3, 3), rnd.matrix(1, 3),
                             oracle::random_exponents(rnd, 3, 2), {1, 2}, cora::detail::Verbatim{});
        oracle::check("polyZonotope", ConPolyZono(P), lean::ConPolyZono::from_poly_zonotope(P));
        const ConZonotope cz = rnd.con_zonotope(3, 5, 2);
        oracle::check("conZonotope", ConPolyZono(cz), lean::ConPolyZono::from_con_zonotope(cz));
        oracle::check("origin", ConPolyZono::origin(3), lean::ConPolyZono::origin(3));
        oracle::check("empty", ConPolyZono::empty(3), lean::ConPolyZono::empty(3));
    });
    // the dimension and the center
    oracle::run("conPolyZono.dim", [](oracle::Random &rnd) {
        const ConPolyZono S = aux_set(rnd);
        oracle::check(S.dim(), lean::ConPolyZono(S).dim());
        oracle::check("center", S.center(), lean::ConPolyZono(S).center());
    });
    // the map and the sums
    oracle::run("conPolyZono.mtimes", [](oracle::Random &rnd) {
        const ConPolyZono S = aux_set(rnd);
        const Tensor M = rnd.matrix(2, 3);
        oracle::check(S.linear_map(M), lean::ConPolyZono(S).linear_map(M));
    });
    oracle::run("conPolyZono.plus", [](oracle::Random &rnd) {
        const ConPolyZono S = aux_set(rnd), T = aux_set(rnd, 1);
        oracle::check(S.plus(T), lean::ConPolyZono(S).plus(lean::ConPolyZono(T)));
    });
    oracle::run("conPolyZono.exactPlus", [](oracle::Random &rnd) {
        const ConPolyZono S = aux_set(rnd);
        const ConPolyZono T(PolyZonotope(rnd.point(3), rnd.matrix(2, 3), rnd.matrix(1, 3),
                                         oracle::random_exponents(rnd, 2, 3), S.ids(),
                                         cora::detail::Verbatim{}),
                            S.constraints());
        oracle::check(S.exact_plus(T), lean::ConPolyZono(S).exact_plus(lean::ConPolyZono(T)));
    });
    // the product, on blocks
    oracle::run("conPolyZono.cartProd", [](oracle::Random &rnd) {
        const ConPolyZono S = aux_set(rnd), T = oracle::con_poly_zono(rnd, {2, 2, 2, 1, 1, 2});
        oracle::check(S.cart_prod(T), lean::ConPolyZono(S).cart_prod(lean::ConPolyZono(T)));
    });
    // the intersection, the combination and the hull
    oracle::run("conPolyZono.and", [](oracle::Random &rnd) {
        const ConPolyZono S = aux_set(rnd), T = aux_set(rnd, 1);
        oracle::check(S.intersect(T), lean::ConPolyZono(S).intersect(lean::ConPolyZono(T)));
    });
    oracle::run("conPolyZono.linComb", [](oracle::Random &rnd) {
        const ConPolyZono S = aux_set(rnd), T = aux_set(rnd, 1);
        oracle::check(S.linear_combination(T),
                      lean::ConPolyZono(S).linear_combination(lean::ConPolyZono(T)));
    });
    oracle::run("conPolyZono.convHull", [](oracle::Random &rnd) {
        // CORALean's hull is of sets without independent generators
        const ConPolyZono S = oracle::con_poly_zono(rnd, {2, 3, 2, 0, 1, 2}),
                          T = oracle::con_poly_zono(rnd, {2, 2, 2, 0, 1, 2});
        oracle::check(S.convex_hull(T), lean::ConPolyZono(S).convex_hull(lean::ConPolyZono(T)));
    });
    // the coordinates
    oracle::run("conPolyZono.project", [](oracle::Random &rnd) {
        const ConPolyZono S = aux_set(rnd);
        oracle::check(S.project({2, 0}).as<ConPolyZono>(), lean::ConPolyZono(S).project({2, 0}));
        oracle::check("lift", S.lift(3, {1, 2, 0}), lean::ConPolyZono(S).lift(3, {1, 2, 0}));
    });
    // the reductions
    oracle::run("conPolyZono.compact", [](oracle::Random &rnd) {
        const ConPolyZono S = aux_set(rnd);
        oracle::check(S.compact(), lean::ConPolyZono(S).compact());
    });
    oracle::run("conPolyZono.reduce", [](oracle::Random &rnd) {
        for (const int order : {1, 2}) {
            const ConPolyZono S = oracle::con_poly_zono(rnd, {2, 6, 3, 3, 1, 2});
            oracle::check(S.reduce(order), lean::ConPolyZono(S).reduce(order));
        }
    });
    // the enclosures and the predicates
    oracle::run("conPolyZono.interval", [](oracle::Random &rnd) {
        const ConPolyZono S = aux_set(rnd);
        oracle::check(S.interval(), lean::ConPolyZono(S).interval());
        oracle::check("zonotope", S.zonotope(), lean::ConPolyZono(S).zonotope());
        oracle::check("polyZonotope", S.poly_zonotope(), lean::ConPolyZono(S).poly_zonotope());
    });
    // the structural equality and the point test
    oracle::run("conPolyZono.isequal", [](oracle::Random &rnd) {
        const ConPolyZono S = aux_set(rnd), T = aux_set(rnd);
        oracle::check(S.isequal(S, 0.0), lean::ConPolyZono(S).isequal(lean::ConPolyZono(S)));
        oracle::check(S.isequal(T, 0.0), lean::ConPolyZono(S).isequal(lean::ConPolyZono(T)));
        oracle::check("representsa", S.representsa("point"),
                      lean::ConPolyZono(S).representsa_point());
    });
    return test::finish("lean con poly zono");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
