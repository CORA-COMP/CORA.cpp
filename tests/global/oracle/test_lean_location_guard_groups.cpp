// test_lean_location_guard_groups - the crossings of the guard hits against the CORALean oracle
//
// The hits of each guard, split where the set indices have a gap, as CORA's aux_groupSets and
// aux_removeGaps: CORA.cpp's priv_guard_groups and the oracle's `location.guardGroups` give the
// same groups in the same order (a split-off part goes to the front). The hits are random pairs
// in the order potInt returns them, guard by guard with ascending set indices.
//
// CORA: hybridDynamics/@location/guardIntersect.m
// CORA authors: Matthias Althoff, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
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
    test::oracle::run("location.guardGroups", [](test::oracle::Random &rnd) {
        std::size_t split = 0; // the groups that came from a gap
        for (int trial = 0; trial < 6; ++trial) {
            // up to three guards, each hit by a random subset of ten sets
            GuardHits hits;
            std::vector<std::vector<int64_t>> pairs;
            for (int64_t g = 0; g < 3; ++g)
                for (int64_t k = 0; k < 10; ++k)
                    if (rnd.uniform(0, 1) < 0.5 + 0.1 * double(g)) {
                        hits.transitions.push_back(g);
                        hits.sets.push_back(k);
                        pairs.push_back({g, k});
                    }
            std::vector<int64_t> mine, theirs;
            const std::vector<GuardGroup> groups = priv_guard_groups(hits);
            for (const GuardGroup &group : groups) {
                mine.push_back(-1); // a separator, so the groups stay apart
                mine.push_back(group.transition);
                mine.insert(mine.end(), group.sets.begin(), group.sets.end());
            }
            for (const std::vector<int64_t> &group : lean::Location::guard_groups(pairs)) {
                theirs.push_back(-1);
                theirs.insert(theirs.end(), group.begin(), group.end());
            }
            test::oracle::check(mine, theirs);
            split += groups.size() > 3 ? groups.size() - 3 : 0;
        }
        test::check(split > 0, "location.guardGroups: some guards split at a gap");
    });
    return test::finish("lean location guardGroups");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
