// test_lean_location_check_flow - checkFlow against the CORALean oracle
//
// The sets whose flow may leave through a hyperplane.
// A set is dropped only if the flow at the center of its projection onto the hyperplane and the
// upper bound of the flow set along the normal are both not positive (`location.checkFlow`), with
// the input set B U + c. The normal is the unit outward normal the caller chose.
//
// ORACLE: hybridDynamics/@location/checkFlow.m
// Authors:       Claude Sonnet 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/location.h"
#include "hybrid_dynamics/location/private/priv.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("location.checkFlow", [](test::oracle::Random &rnd) {
        int kept = 0, total = 0;
        for (const int64_t n : {2, 3}) {
            for (int trial = 0; trial < 6; ++trial) {
                // a flow with an input and a constant, and a hyperplane x'n = d with a unit normal
                const Tensor A = rnd.matrix(n, n), B = rnd.matrix(n, 2), c = rnd.vector(n);
                const Zonotope U = rnd.zonotope(2, 2);
                Tensor normal = rnd.vector(n);
                normal = normal / normal.square().sum().sqrt().item();
                const double d = rnd.uniform(-1, 1);
                const Location loc("l", Polytope::inf(n), {}, LinearSys(A, B), c);
                std::vector<Zonotope> sets;
                std::vector<Set> reach;
                for (int k = 0; k < 6; ++k) {
                    sets.push_back(rnd.zonotope(n, n + 2));
                    reach.emplace_back(sets.back());
                }
                const Zonotope V = priv_state_input(loc, U);
                std::vector<int64_t> mine, theirs;
                // the steps of one set at a time: a hyperplane through the set carries its flow
                for (std::size_t k = 0; k < sets.size(); ++k)
                    if (!priv_check_flow_outward(loc, normal, d, {reach[k]}, V).empty())
                        mine.push_back(int64_t(k));
                theirs = lean::Location::check_flow({A, B, c, U}, normal, d, sets);
                test::oracle::check(mine, theirs);
                kept += int(theirs.size());
                total += int(sets.size());
            }
        }
        test::check(kept > 0 && kept <= total, "location.checkFlow: some sets are kept");
    });
    return test::finish("lean location checkFlow");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
