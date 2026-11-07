// test_lean_con_zonotope_maps - plus, mtimes, origin, project, lift and the predicates of
//
// ConZonotope
//
// Bit for bit where CORALean builds the same set: origin, plus, mtimes, project, lift. isequal and
// the two proofs (isIntersecting: a proved disjointness; representsa: a proved emptiness) agree
// where CORALean decides. reduce, quadMap and split: test_lean_con_zonotope_enclosure.
//
// ORACLE: contSet/@conZonotope/origin.m
// ORACLE: contSet/@conZonotope/plus.m
// ORACLE: contSet/@conZonotope/mtimes.m
// ORACLE: contSet/@conZonotope/project.m
// ORACLE: contSet/@conZonotope/lift_.m
// ORACLE: contSet/@conZonotope/isequal.m
// ORACLE: contSet/@conZonotope/isIntersecting_.m
// ORACLE: contSet/@conZonotope/representsa_.m
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "cora/global/oracle/con_zonotope.h"
#include "oracle_testing.h"

using namespace cora;
namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    oracle::run("conZonotope.origin", [](oracle::Random &) {
        oracle::check(ConZonotope::origin(3), lean::ConZonotope::origin(3));
    });
    oracle::run("conZonotope.plus", [](oracle::Random &rnd) {
        const ConZonotope a = rnd.con_zonotope(3, 5, 2), b = rnd.con_zonotope(3, 4, 1);
        oracle::check(a.plus(b), lean::ConZonotope(a).plus(b));
    });
    // the maps are exact: the same bits at nearest
    oracle::run("conZonotope.mtimes", [](oracle::Random &rnd) {
        const ConZonotope a = rnd.con_zonotope(3, 5, 2);
        const Tensor M = rnd.matrix(2, 3);
        oracle::check(a.linear_map(M), lean::ConZonotope(a).mtimes(M));
    });
    oracle::run("conZonotope.project", [](oracle::Random &rnd) {
        const ConZonotope a = rnd.con_zonotope(4, 5, 2);
        oracle::check(a.project({3, 1}).as<ConZonotope>(), lean::ConZonotope(a).project({3, 1}));
    });
    oracle::run("conZonotope.lift", [](oracle::Random &rnd) {
        const ConZonotope a = rnd.con_zonotope(3, 5, 2);
        oracle::check(a.lift(3, {1, 2, 0}), lean::ConZonotope(a).lift(3, {1, 2, 0}));
    });
    // the predicates agree where CORALean decides: equal sets, and a proof of disjointness
    oracle::run("conZonotope.isequal", [](oracle::Random &rnd) {
        const ConZonotope a = rnd.con_zonotope(3, 5, 2), b = rnd.con_zonotope(3, 5, 2);
        oracle::check(a.isequal(a), lean::ConZonotope(a).isequal(a));
        oracle::check(a.isequal(b), lean::ConZonotope(a).isequal(b));
    });
    oracle::run("conZonotope.isIntersecting", [](oracle::Random &rnd) {
        const ConZonotope a = rnd.con_zonotope(3, 5, 2);
        const ConZonotope near = a.plus(Tensor::like(a.c(), {0.1, 0.0, 0.0}, {3, 1}));
        const ConZonotope far = a.plus(Tensor::like(a.c(), {1000.0, 0.0, 0.0}, {3, 1}));
        oracle::check(a.is_intersecting(near), lean::ConZonotope(a).is_intersecting(near));
        oracle::check(a.is_intersecting(far), lean::ConZonotope(a).is_intersecting(far));
    });
    oracle::run("conZonotope.representsa", [](oracle::Random &rnd) {
        const ConZonotope a = rnd.con_zonotope(3, 5, 2);
        // the offset of the first row is out of the box: no factor vector
        const Tensor b = a.b() + Tensor::like(a.b(), {5.0, 0.0}, {2, 1});
        const ConZonotope none(a.c(), a.generators(), a.A(), b);
        oracle::check(a.representsa("emptySet"), lean::ConZonotope(a).representsa_empty());
        oracle::check(none.representsa("emptySet"), lean::ConZonotope(none).representsa_empty());
    });
    return test::finish("lean con zonotope maps");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
