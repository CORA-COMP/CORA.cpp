// test_lean_zono_bundle_maps - the maps and sums of ZonoBundle against the CORALean oracle
//
// Bit for bit: mtimes, project, reduce and replace map every member on its own; plus is
// the sum of every pair (the paper's, not CORA's box of the second bundle); origin; zonotope is the
// member of least volume. quadMap is inside the oracle's enclosure (the generators are laid out
// differently).
//
// ORACLE: contSet/@zonoBundle/mtimes.m
// ORACLE: contSet/@zonoBundle/origin.m
// ORACLE: contSet/@zonoBundle/plus.m
// ORACLE: contSet/@zonoBundle/project.m
// ORACLE: contSet/@zonoBundle/quadMap.m
// ORACLE: contSet/@zonoBundle/reduce.m
// ORACLE: contSet/@zonoBundle/replace.m
// ORACLE: contSet/@zonoBundle/zonotope.m
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zono_bundle/zono_bundle.h"
#include "cora/global/oracle/zono_bundle.h"
#include "oracle_testing.h"

using namespace cora;
namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    oracle::run("zonoBundle.mtimes", [](oracle::Random &rnd) {
        const ZonoBundle a = rnd.zono_bundle(3, 2, 4);
        const Tensor M = rnd.matrix(2, 3);
        oracle::check(a.linear_map(M), lean::ZonoBundle(a).linear_map(M));
    });
    oracle::run("zonoBundle.plus", [](oracle::Random &rnd) {
        const ZonoBundle a = rnd.zono_bundle(3, 2, 4), b = rnd.zono_bundle(3, 3, 2);
        oracle::check(a.plus(b), lean::ZonoBundle(a).plus(b));
    });
    // the maps that take dimensions, matrices or an order
    oracle::run("zonoBundle.project", [](oracle::Random &rnd) {
        const ZonoBundle a = rnd.zono_bundle(4, 2, 4);
        const std::vector<int64_t> dims{3, 1};
        oracle::check(Set(a)->project(dims).as<ZonoBundle>(), lean::ZonoBundle(a).project(dims));
    });
    oracle::run("zonoBundle.quadMap", [](oracle::Random &rnd) {
        const ZonoBundle a = rnd.zono_bundle(3, 2, 3);
        const std::vector<Tensor> Q{rnd.matrix(3, 3), rnd.matrix(3, 3)};
        const std::vector<lean::Tensor> theirs_Q{lean::Tensor(Q[0]), lean::Tensor(Q[1])};
        // the oracle lays the new generators out differently: an enclosure of CORA.cpp's set
        oracle::check_enclosed("zonoBundle.quadMap", a.quad_map(Q),
                               lean::ZonoBundle(a).quad_map(theirs_Q));
    });
    oracle::run("zonoBundle.reduce", [](oracle::Random &rnd) {
        const ZonoBundle a = rnd.zono_bundle(2, 3, 7);
        oracle::check(a.reduce(2), lean::ZonoBundle(a).reduce(2));
    });
    // replacing a member, the origin and the member of least volume
    oracle::run("zonoBundle.replace", [](oracle::Random &rnd) {
        const ZonoBundle a = rnd.zono_bundle(3, 3, 2);
        const Zonotope Z = rnd.zonotope(3, 4);
        oracle::check(a.replace(1, Z), lean::ZonoBundle(a).replace(1, Z));
    });
    oracle::run("zonoBundle.origin", [](oracle::Random &) {
        oracle::check(ZonoBundle::origin(3), lean::ZonoBundle::origin(3));
    });
    oracle::run("zonoBundle.zonotope", [](oracle::Random &rnd) {
        const ZonoBundle a = rnd.zono_bundle(3, 3, 4);
        oracle::check(a.zonotope(), lean::ZonoBundle(a).zonotope());
    });
    return test::finish("lean zono bundle maps");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
