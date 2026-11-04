// test_lean_zono_bundle - the operations of ZonoBundle that the CORALean oracle serves
//
// Bit for bit where CORALean builds the same set: and, cartProd, empty, enlarge, interval,
// isBounded, isIntersecting (CORA's "approx": the oracle tests the boxes), isequal (members) and
// lift. The constrained zonotope is checked as CORALean's form of the same set (inequality rows for
// the agreement of the points, as for conZonotope.and). enclose is not compared: CORALean's is the
// generic route (all combinations of members), CORA's is member by member, so the sets differ.
//
// ORACLE: contSet/@zonoBundle/and_.m
// ORACLE: contSet/@zonoBundle/cartProd_.m
// ORACLE: contSet/@zonoBundle/conZonotope.m
// ORACLE: contSet/@zonoBundle/empty.m
// ORACLE: contSet/@zonoBundle/enlarge.m
// ORACLE: contSet/@zonoBundle/interval.m
// ORACLE: contSet/@zonoBundle/isBounded.m
// ORACLE: contSet/@zonoBundle/isIntersecting_.m
// ORACLE: contSet/@zonoBundle/isequal.m
// ORACLE: contSet/@zonoBundle/lift_.m
//
// CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger, Adrian Kulmburg, Tobias
// Ladner Authors:       Claude Sonnet 5.5 Written:       04-November-2026 Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zono_bundle/zono_bundle.h"
#include "cora/global/oracle/zono_bundle.h"
#include "oracle_testing.h"

using namespace cora;
namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The rows [from, to) of a matrix.
Tensor aux_rows(const Tensor &T, int64_t from, int64_t to) { return T.slice(0, from, to); }

/// The bundle with every member shifted by `offset` (n, 1): the same shape, far from the original.
ZonoBundle aux_shifted(const ZonoBundle &bundle, const Tensor &offset) {
    std::vector<Zonotope> members;
    for (const Zonotope &Z : bundle.zonotopes())
        members.emplace_back(Z.center() + offset, Z.generators());
    return ZonoBundle(std::move(members));
}

/// The constrained zonotope of a bundle of two members against CORALean's form: the same center and
/// generators, and the agreement of the points as the rows c1 + G1 x1 - c2 - G2 x2 <= 0 and >= 0
/// where CORA has equalities (as for conZonotope.and).
void aux_check_con_zonotope(const ConZonotope &mine, const lean::ConZonotope &theirs) {
    if (oracle::mode() == oracle::Mode::sound)
        return oracle::check_enclosed("conZonotope", mine, theirs);
    const ConZonotope their = theirs.nominal();
    const auto [A, b_ineq] = theirs.inequalities();
    const int64_t pe = their.A().shape()[0], n = mine.dim();
    oracle::check_exact("conZonotope.c", mine.c(), their.c());
    oracle::check_exact("conZonotope.G", mine.generators(), their.generators());
    oracle::check_exact("conZonotope.A", aux_rows(mine.A(), pe, pe + n), aux_rows(A, 0, n));
    oracle::check_exact("conZonotope.b", aux_rows(mine.b(), pe, pe + n), aux_rows(b_ineq, 0, n));
    oracle::check_exact("conZonotope.-A", -aux_rows(mine.A(), pe, pe + n), aux_rows(A, n, 2 * n));
    oracle::check_exact("conZonotope.-b", -aux_rows(mine.b(), pe, pe + n),
                        aux_rows(b_ineq, n, 2 * n));
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    oracle::run("zonoBundle.and", [](oracle::Random &rnd) {
        const ZonoBundle a = rnd.zono_bundle(3, 2, 4), b = rnd.zono_bundle(3, 3, 2);
        oracle::check(a.intersect(b), lean::ZonoBundle(a).intersect(b));
    });
    oracle::run("zonoBundle.cartProd", [](oracle::Random &rnd) {
        const ZonoBundle a = rnd.zono_bundle(3, 2, 4), b = rnd.zono_bundle(2, 3, 2);
        oracle::check(a.cart_prod(b), lean::ZonoBundle(a).cart_prod(b));
    });
    oracle::run("zonoBundle.conZonotope", [](oracle::Random &rnd) {
        const ZonoBundle a = rnd.zono_bundle(3, 2, 4);
        aux_check_con_zonotope(a.con_zonotope(), lean::ZonoBundle(a).con_zonotope());
    });
    oracle::run("zonoBundle.empty", [](oracle::Random &) {
        oracle::check(ZonoBundle::empty(3), lean::ZonoBundle::empty(3));
    });
    oracle::run("zonoBundle.enlarge", [](oracle::Random &rnd) {
        const ZonoBundle a = rnd.zono_bundle(3, 3, 4);
        const Tensor factor = rnd.vector(3);
        oracle::check(Set(a)->enlarge(factor).as<ZonoBundle>(),
                      lean::ZonoBundle(a).enlarge(factor));
    });
    oracle::run("zonoBundle.interval", [](oracle::Random &rnd) {
        const ZonoBundle a = rnd.zono_bundle(3, 3, 4);
        oracle::check(a.interval(), lean::ZonoBundle(a).interval());
    });
    oracle::run("zonoBundle.isBounded", [](oracle::Random &rnd) {
        const ZonoBundle a = rnd.zono_bundle(3, 2, 4);
        oracle::check(a.is_bounded(), lean::ZonoBundle(a).is_bounded());
    });
    oracle::run("zonoBundle.isIntersecting", [](oracle::Random &rnd) {
        const ZonoBundle a = rnd.zono_bundle(3, 2, 4), b = rnd.zono_bundle(3, 2, 3);
        const ZonoBundle far = aux_shifted(b, Tensor::from_data({100.0, 100.0, 100.0}, {3, 1}));
        oracle::check(a.is_intersecting(b, "approx", 0.0), lean::ZonoBundle(a).is_intersecting(b));
        oracle::check(a.is_intersecting(far, "approx", 0.0),
                      lean::ZonoBundle(a).is_intersecting(far));
    });
    oracle::run("zonoBundle.isequal", [](oracle::Random &rnd) {
        const ZonoBundle a = rnd.zono_bundle(3, 2, 4), b = rnd.zono_bundle(3, 2, 4);
        oracle::check(a.isequal(a), lean::ZonoBundle(a).isequal(a));
        oracle::check(a.isequal(b), lean::ZonoBundle(a).isequal(b));
    });
    oracle::run("zonoBundle.lift", [](oracle::Random &rnd) {
        const ZonoBundle a = rnd.zono_bundle(3, 2, 4);
        const std::vector<int64_t> dims{2, 0, 1};
        oracle::check(a.lift(3, dims), lean::ZonoBundle(a).lift(3, dims));
    });
    return test::finish("lean zono bundle");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
