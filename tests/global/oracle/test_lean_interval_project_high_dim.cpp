// test_lean_interval_project_high_dim - Interval::project_high_dim against the CORALean oracle
//
// ORACLE: contSet/@interval/projectHighDim_.m
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.projectHighDim", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5}) {
            const Interval I = rnd.box(n);
            std::vector<int64_t> proj(n);
            for (int j = 0; j < n; ++j) proj[j] = 2 * (n - 1 - j) + 1;
            test::oracle::check(I.project_high_dim(2 * n + 1, proj),
                                lean::Interval(I).project_high_dim(2 * n + 1, proj));
        }
    });
    return test::finish("lean interval project_high_dim");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
