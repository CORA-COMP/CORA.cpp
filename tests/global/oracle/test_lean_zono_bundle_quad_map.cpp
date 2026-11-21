// test_lean_zono_bundle_quad_map - ZonoBundle::quad_map against the CORALean oracle, bit for bit
//
// The oracle's `zonoBundle.quadMapCora` maps each member by the quadratic map of CORA (the one
// that Zonotope::quad_map is compared with); the result is the bundle of the mapped members.
//
// ORACLE: contSet/@zonoBundle/quadMap.m
// Authors:       Claude Sonnet 5.5
// Written:       21-November-2026
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
    oracle::run("zonoBundle.quadMapCora", [](oracle::Random &rnd) {
        for (const int64_t n : {2, 3}) {
            const ZonoBundle a = rnd.zono_bundle(n, 2, 3);
            std::vector<Tensor> Q;
            std::vector<lean::Tensor> theirs_Q;
            for (int64_t i = 0; i < n; ++i) {
                Q.push_back(rnd.matrix(n, n));
                theirs_Q.emplace_back(Q.back());
            }
            oracle::check(a.quad_map(Q), lean::ZonoBundle(a).quad_map_cora(theirs_Q));
        }
    });
    return test::finish("lean zono_bundle quad_map");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
