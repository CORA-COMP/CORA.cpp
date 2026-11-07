// test_lean_zono_bundle_quad_map_enclosure - ZonoBundle::quad_map inside the CORALean oracle's
//
// result, member by member
//
// The oracle maps each member through its own (wider) enclosure, with the new generators laid out
// differently from CORA.cpp, so no bit-exact test exists; the box of each member of CORA.cpp's
// result lies inside the box of the oracle's member.
//
// CORA: contSet/@zonoBundle/quadMap.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       07-November-2026
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
    oracle::run("zonoBundle.quadMap", [](oracle::Random &rnd) {
        const ZonoBundle a = rnd.zono_bundle(3, 2, 3);
        const std::vector<Tensor> Q{rnd.matrix(3, 3), rnd.matrix(3, 3)};
        const std::vector<lean::Tensor> theirs_Q{lean::Tensor(Q[0]), lean::Tensor(Q[1])};
        const ZonoBundle mine = a.quad_map(Q);
        const lean::ZonoBundle theirs = lean::ZonoBundle(a).quad_map(theirs_Q);
        test::check(mine.parallel_sets() == int64_t(theirs.zonotopes().size()),
                    "zonoBundle.quadMap: one member per member");
        // member by member, as the boxes they span
        for (std::size_t i = 0; i < theirs.zonotopes().size(); ++i)
            oracle::check_inside(mine.zonotopes()[i].interval(), theirs.zonotopes()[i].interval());
    });
    return test::finish("lean zono_bundle quad_map");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
