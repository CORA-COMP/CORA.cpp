// test_lean_zonotope_project - Zonotope::project against the CORALean oracle
//
// ORACLE: contSet/@zonotope/project.m
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/oracle/zonotope.h"
#include "oracle_testing.h"

using namespace cora;


namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    oracle::run("zonotope.project", [](oracle::Random &rnd) {
        for (const auto &[n, m] : {std::pair{2, 3}, {5, 30}, {20, 50}}) {
            const Zonotope Z = rnd.zonotope(n, m);
            // the first dimension, two in reverse order, and all of them
            std::vector<std::vector<int64_t>> choices{{0}, {int64_t(n - 1), 0}};
            std::vector<int64_t> all(n);
            for (int i = 0; i < n; ++i) all[i] = i;
            choices.push_back(all);
            for (const std::vector<int64_t> &dims : choices)
                oracle::check(Z.project(dims).as<Zonotope>(), lean::Zonotope(Z).project(dims));
        }
    });
    return test::finish("lean zonotope project");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
