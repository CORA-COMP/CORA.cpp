// test_lean_zonotope_mink_diff_enclosure - Zonotope::mink_diff against the CORALean oracle, as an inner
// approximation
//
// For generators that are aligned (the subtrahend's are those of the minuend scaled by one factor)
// CORA's "exact" scales the generators of the minuend; the oracle's `aligned` method (Althoff,
// Prop. 5) gives the generators g1 - g2 shrunk by rho < 1 and certified, so a subset of the difference,
// or nothing where its check fails. The check is that the oracle's set lies in CORA.cpp's. The two do
// not compute the same set (rho), so there is no ORACLE tag.

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/oracle/zonotope.h"
#include "oracle_testing.h"

using namespace cora;
namespace oracle = test::oracle;

int main() {
    if (!oracle::available()) return 0;
    oracle::run("zonotope.minkDiff", [](oracle::Random &rnd) {
        int certified = 0;
        for (const int n : {2, 3}) {
            for (int trial = 0; trial < 3; ++trial) {
                const Zonotope Z = rnd.zonotope(n, n);
                const Tensor G2 = Z.generators() * rnd.uniform(0.2, 0.6);
                const Zonotope Z2(rnd.vector(n) * 0.1, G2);
                const std::optional<lean::Zonotope> theirs =
                    lean::Zonotope(Z).mink_diff_aligned(lean::Zonotope(Z2), 0.99);
                if (!theirs) continue;
                ++certified;
                const Zonotope mine = Z.mink_diff(Z2, "exact");
                test::check(mine.contains(theirs->gather().first, "exact", 1e-9),
                            "zonotope.minkDiff: the oracle's difference lies in CORA.cpp's");
            }
        }
        test::check(certified > 0, "zonotope.minkDiff: the oracle certifies some differences");
    });
    return test::finish("lean zonotope mink_diff enclosure");
}
