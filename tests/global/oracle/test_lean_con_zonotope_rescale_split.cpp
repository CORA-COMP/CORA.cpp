// test_lean_con_zonotope_rescale_split - rescale, split, intersect_strip and reduce of ConZonotope
//
// against the CORALean oracle, bit for bit
//
// rescale scales at the factor ranges of its linear programs, which are sent as the certificate
// (ConZonotope.rescale); split cuts CORA.cpp's box of the set at (inf + sup) / 2 and intersects
// (ConZonotope.splitWith at that box); intersect_strip at a given gain builds G - L (C G) and
// L diag(phi) (ConZonotope.intersectStrip); reduce is Girard's on [G; A] (ConZonotope.reduce).
// The programs themselves (the ranges, the box) and the gains of the methods have no oracle op.
//
// ORACLE: contSet/@conZonotope/rescale.m
// ORACLE: contSet/@conZonotope/split.m
// ORACLE: contSet/@conZonotope/intersectStrip.m
// ORACLE: contSet/@conZonotope/reduce.m
// ORACLE: contSet/@conZonotope/dim.m
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "cora/global/oracle/con_zonotope.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

using namespace cora;
namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    // the scaling at the ranges of the linear programs
    oracle::run("conZonotope.rescale", [](oracle::Random &rnd) {
        for (const int64_t m : {4, 6}) {
            const ConZonotope cz = rnd.con_zonotope(3, m, 2);
            const auto [lo, hi] = priv_factor_ranges(priv_host_con_zonotope(cz));
            const Tensor L = Tensor::from_data(lo, {m}), H = Tensor::from_data(hi, {m});
            oracle::check(cz.rescale("exact"), lean::ConZonotope(cz).rescale(L, H));
        }
    });
    // the halves of CORA.cpp's box, in every coordinate
    oracle::run("conZonotope.split", [](oracle::Random &rnd) {
        const ConZonotope cz = rnd.con_zonotope(3, 5, 2);
        const lean::Interval I(cz.interval());
        for (int64_t d = 0; d < 3; ++d) {
            const auto [a, b] = cz.split(d);
            const auto [la, lb] = lean::ConZonotope(cz).split(d, I);
            oracle::check("split.lower", a, la);
            oracle::check("split.upper", b, lb);
        }
    });
    // the strips at a given gain
    oracle::run("conZonotope.intersectStrip", [](oracle::Random &rnd) {
        for (const int64_t k : {1, 2}) {
            const ConZonotope cz = rnd.con_zonotope(3, 5, 2);
            const Tensor C = rnd.matrix(k, 3), L = rnd.matrix(3, k);
            const Tensor phi = rnd.point(k).abs() + 0.5, y = rnd.point(k);
            oracle::check(cz.intersect_strip(C, phi, y, L),
                          lean::ConZonotope(cz).intersect_strip(C, phi, y, L));
        }
    });
    // Girard's reduction of the lifted zonotope, and the dimension
    oracle::run("conZonotope.reduce", [](oracle::Random &rnd) {
        for (const int order : {1, 2}) {
            const ConZonotope cz = rnd.con_zonotope(2, 9, 2);
            oracle::check(cz.reduce(order), lean::ConZonotope(cz).reduce(order));
        }
    });
    oracle::run("conZonotope.dim", [](oracle::Random &rnd) {
        const ConZonotope cz = rnd.con_zonotope(3, 5, 2);
        oracle::check(cz.dim(), lean::ConZonotope(cz).dim());
    });
    return test::finish("lean con zonotope rescale split");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
