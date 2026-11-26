// test_lean_con_poly_zono_split - ConPolyZono's split, get_subset and the factor splits
//
// against the CORALean oracle, bit for bit
//
// split cuts CORA.cpp's box of the set at (inf + sup) / 2 and intersects each half's zonotope
// (ConPolyZono.splitWith at that box); get_subset substitutes the factor in the generators and the
// rows, every binomial term a column (ConPolyZono.getSubset); split_dep_factor and
// split_longest_gen are get_subset at [-1, 0] and [0, 1] (the longest factor by longestFactor).
//
// ORACLE: contSet/@conPolyZono/split.m
// ORACLE: contSet/@conPolyZono/getSubset.m
// ORACLE: contSet/@conPolyZono/splitDepFactor.m
// ORACLE: contSet/@conPolyZono/splitLongestGen.m
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_poly_zono_oracle_testing.h"
#include "cora/global/oracle/con_poly_zono.h"
#include "cora/global/oracle/interval.h"

using namespace cora;
namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    // the halves of CORA.cpp's box, in every coordinate
    oracle::run("conPolyZono.split", [](oracle::Random &rnd) {
        const ConPolyZono S = oracle::con_poly_zono(rnd, {2, 3, 2, 1, 1, 2});
        const lean::Interval I(S.interval());
        for (int64_t d = 0; d < 2; ++d) {
            const auto [a, b] = S.split(d);
            const auto [la, lb] = lean::ConPolyZono(S).split(d, I);
            oracle::check("split.lower", a, la);
            oracle::check("split.upper", b, lb);
        }
    });
    // a factor restricted to a sub-box, in the generators and the rows
    oracle::run("conPolyZono.getSubset", [](oracle::Random &rnd) {
        const ConPolyZono S = oracle::con_poly_zono(rnd, {3, 4, 3, 1, 2, 3});
        const Interval val(Tensor::from_data({-0.25}, {1}), Tensor::from_data({0.75}, {1}));
        oracle::check(S.get_subset(2, val), lean::ConPolyZono(S).get_subset(2, 0.25, 0.5));
    });
    // the halves of a factor, and of the longest one
    oracle::run("conPolyZono.splitDepFactor", [](oracle::Random &rnd) {
        const ConPolyZono S = oracle::con_poly_zono(rnd, {3, 4, 3, 1, 2, 3});
        const auto [a, b] = S.split_dep_factor(1);
        const auto [la, lb] = lean::ConPolyZono(S).split_dep_factor(1);
        oracle::check("splitDepFactor.lower", a, la);
        oracle::check("splitDepFactor.upper", b, lb);
    });
    oracle::run("conPolyZono.splitLongestGen", [](oracle::Random &rnd) {
        const ConPolyZono S = oracle::con_poly_zono(rnd, {3, 4, 3, 1, 2, 3});
        const auto [a, b] = S.split_longest_gen();
        const auto [la, lb] = lean::ConPolyZono(S).split_longest_gen();
        oracle::check("splitLongestGen.lower", a, la);
        oracle::check("splitLongestGen.upper", b, lb);
    });
    return test::finish("lean con poly zono split");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
