// test_lean_con_zonotope_certified_enclosure - ConZonotope::contains (a point) and is_full_dim
//
// against the CORALean oracle, as proofs that are never contradicted
//
// The oracle's answers are sufficient only (`ConZonotope.containsPoint`, `isFullDim`): a true
// proves the point inside, or an interior point of the set, by a right inverse of G and a ball
// around the center that is not the whole set, and a set with an equality row is never proved. So a
// proof of the oracle must agree with CORA.cpp, and some are proved; the answers are not equal
// otherwise (a point inside but far from the center is not proved), so there is no ORACLE tag. The
// sets are zonotopes as constrained zonotopes, in both modes.
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
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
    oracle::run("conZonotope.contains", [](oracle::Random &rnd) {
        int proved = 0;
        for (const auto &[n, m] : {std::pair{2, 2}, {3, 3}, {2, 3}}) {
            const Zonotope Z = rnd.zonotope(n, m);
            const ConZonotope cz(Z);
            const lean::ConZonotope theirs = lean::ConZonotope::from_zonotope(lean::Zonotope(Z));
            // the center, a point near it, one inside, and one far outside
            const Tensor near = Z.center() + Z.generators().matmul(rnd.vector(m)) * 0.01,
                         inside = Z.center() + Z.generators().matmul(rnd.vector(m)) * 0.2,
                         outside = Z.center() + Z.generators().matmul(rnd.vector(m).sign()) * 3.0;
            for (const Tensor &p : {Z.center(), near, inside, outside}) {
                const bool theirs_proved = theirs.contains(p);
                proved += theirs_proved;
                oracle::check_enclosed(oracle::op_name(), cz.contains(p),
                                       lean::Certified{theirs_proved, theirs_proved});
            }
        }
        test::check(proved > 0, "conZonotope.contains: the oracle proves some points");
    });
    oracle::run("conZonotope.isFullDim", [](oracle::Random &rnd) {
        int proved = 0;
        for (const auto &[n, m] : {std::pair{2, 3}, {3, 3}, {3, 2}, {4, 2}}) {
            const Zonotope Z = rnd.zonotope(n, m);
            const bool theirs = lean::ConZonotope::from_zonotope(lean::Zonotope(Z)).is_full_dim();
            proved += theirs;
            // the oracle proves the interior
            oracle::check_enclosed(oracle::op_name(), ConZonotope(Z).is_full_dim(),
                                   lean::Certified{theirs, theirs});
        }
        test::check(proved > 0, "conZonotope.isFullDim: the oracle proves some interiors");
    });
    return test::finish("lean con zonotope certified enclosure");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
