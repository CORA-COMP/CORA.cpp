// test_lean_location_pot_int - the guards a set may meet against the CORALean oracle
//
// The (transition, set) pairs of CORA.cpp's potInt are the oracle's `location.potInt`: a set meets
// a guard unless a row of the guard lies strictly beyond its support function. The transitions into
// the final location are skipped, and the pairs come guard by guard.
//
// ORACLE: hybridDynamics/@location/potInt.m
// Authors:       Claude Sonnet 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/location.h"
#include "hybrid_dynamics/location/private/priv.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A guard of p random rows, each offset near the lowest value of a set of the batch along it: some
/// rows cut the sets and some lie beyond them.
Polytope aux_guard(test::oracle::Random &rnd, const std::vector<Zonotope> &sets, int64_t n,
                   int64_t p) {
    const Tensor A = rnd.matrix(p, n);
    std::vector<double> b(static_cast<std::size_t>(p));
    for (int64_t i = 0; i < p; ++i) {
        const Zonotope &S = sets[std::size_t(i) % sets.size()];
        const Tensor row = A.slice(-2, i, i + 1).matrix_transpose();
        b[std::size_t(i)] = -S.support_function(-row).item() + rnd.uniform(-1, 1);
    }
    return {A, Tensor::from_data(b, {p, 1})};
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("location.potInt", [](test::oracle::Random &rnd) {
        std::size_t hit = 0, scanned = 0; // the pairs found, and those of the two scanned guards
        for (const int64_t n : {2, 3}) {
            std::vector<Zonotope> sets;
            std::vector<Set> reach;
            for (int k = 0; k < 6; ++k) {
                sets.push_back(rnd.zonotope(n, n + 2));
                reach.emplace_back(sets.back());
            }
            // three guards into the locations 1, 2 and 3; the transitions into 2 are skipped
            std::vector<Polytope> guards;
            std::vector<Transition> transitions;
            const std::vector<int64_t> targets = {1, 2, 3};
            for (const int64_t target : targets) {
                guards.push_back(aux_guard(rnd, sets, n, 2 + target % 2));
                transitions.emplace_back(guards.back(), LinearReset(rnd.matrix(n, n)), target);
            }
            const Location loc("l", Polytope::inf(n), transitions, LinearSys(rnd.matrix(n, n)));
            LocationReachSet R;
            R.time_interval = reach;
            R.time_point = reach;
            const GuardHits hits = priv_pot_int(loc, R, {2});
            // the pairs, flattened
            std::vector<int64_t> mine, theirs;
            for (std::size_t k = 0; k < hits.sets.size(); ++k) {
                mine.push_back(hits.transitions[k]);
                mine.push_back(hits.sets[k]);
            }
            for (const std::vector<int64_t> &pair :
                 lean::Location::pot_int(guards, targets, 2, sets))
                theirs.insert(theirs.end(), pair.begin(), pair.end());
            test::oracle::check(mine, theirs);
            hit += mine.size() / 2;
            scanned += 2 * sets.size();
        }
        test::check(hit > 0 && hit < scanned,
                    "location.potInt: some sets meet a guard and some do not");
    });
    return test::finish("lean location potInt");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
