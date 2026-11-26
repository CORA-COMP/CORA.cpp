// test_con_poly_zono_split_longest_gen - the halves along the longest factor cover x0
//
// CORA: contSet/@conPolyZono/splitLongestGen.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_poly_zono_testing.h"

using namespace cora;
using namespace test::cpz;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &bk) {
        const ConPolyZono S = known();
        const std::vector<double> x0 = point(S, kA0, {kBeta0});
        const auto [l1, l2] = S.split_longest_gen();
        check(in_box(l1, x0) || in_box(l2, x0), bk + ": split_longest_gen");
    });
    {
        ChecksScope off(false);
        (void)known().split_longest_gen();
    }
    return test::finish("con_poly_zono split_longest_gen");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
