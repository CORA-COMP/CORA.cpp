// benchmark_hybrid - the shared parts of the ARCH-COMP AFF hybrid benchmarks
//
// A hybrid instance is the automaton, what reach is given, the specifications and the verdict of
// MATLAB CORA. run_hybrid reaches it, decides it (the specifications during reach, then the
// script's own check of the sets, if any) and prints the line 'benchmark,instance,result,time'
// of the original scripts, with the number of branches on stderr. The helpers write the models'
// matrices row by row, as the .m files do.
//
// Syntax:     cora::bench::run_hybrid(instance);
// See also:   benchmark_hybrid_reach_arch23_rendezvous.cpp, hybrid_automaton.h
//
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "benchmark_run.h"
#include "cora/cora.h"

#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::bench {

/// A matrix of `rows` rows from its entries row by row, as MATLAB writes it.
inline Tensor matrix(int64_t rows, int64_t cols, const std::vector<double> &entries) {
    return Tensor::from_data(entries, {rows, cols});
}

/// A column vector.
inline Tensor column(const std::vector<double> &entries) {
    return Tensor::from_data(entries, {int64_t(entries.size()), 1});
}

/// CORA's polytope([], [], a, b): the hyperplane a'x = b.
inline Polytope hyperplane(const std::vector<double> &a, double b) {
    const auto n = int64_t(a.size());
    return Polytope(Tensor::zeros({0, n}), Tensor::zeros({0, 1}), matrix(1, n, a), column({b}));
}

/// An instance: the automaton and its problem, the specifications, a check of the sets after
/// reach (the scripts that verify afterwards), and the verdict of MATLAB CORA.
struct HybridInstance {
    std::string benchmark, label;
    HybridAutomaton HA;
    HybridReachProblem problem;
    alg::HybridStandard alg;
    std::vector<HybridSpecification> specs = {};
    std::function<bool(const HybridReachSet &)> check = nullptr;
    bool expected = true;
};

/// Every time-interval set of every branch (CORA's query(R, 'reachSet')).
inline std::vector<Set> all_sets(const HybridReachSet &R) {
    std::vector<Set> sets;
    for (const LocationReachSet &b : R.branches)
        sets.insert(sets.end(), b.time_interval.begin(), b.time_interval.end());
    return sets;
}

/// Reaches and decides the instance and prints 'benchmark,instance,result,time'; the time is the
/// wall clock of reach and of the check.
inline bool run_hybrid(const HybridInstance &inst) {
    const auto start = std::chrono::steady_clock::now();
    const HybridReachSet R = inst.HA.reach(inst.problem, inst.alg, inst.specs);
    // the script's own check of the sets, after reach
    const bool verified = R.verified && (!inst.check || inst.check(R));
    const std::chrono::duration<double> wall = std::chrono::steady_clock::now() - start;
    print_line(inst.benchmark, inst.label, verified, wall.count());
    std::size_t sets = 0;
    for (const LocationReachSet &b : R.branches) sets += b.time_interval.size();
    std::cerr << "  " << R.branches.size() << " branches, " << sets
              << " time-interval sets  (MATLAB CORA: " << inst.expected << ")" << std::endl;
    // with CORACPP_BENCHMARK_VERBOSE, each branch: its location, parent, sets and times
    if (std::getenv("CORACPP_BENCHMARK_VERBOSE"))
        for (const LocationReachSet &b : R.branches)
            std::cerr << "    location " << b.location << " (parent " << b.parent
                      << "): " << b.time_interval.size() << " sets from "
                      << b.time_point_times.front()[0] << " to " << b.time_point_times.back()[1]
                      << std::endl;
    return verified == inst.expected;
}

/// The instances named on the command line, else all of the family (run_family).
inline int run_hybrid_family(int argc, char **argv, const std::vector<std::string> &family,
                             const std::function<HybridInstance(const std::string &)> &make) {
    return run_family(argc, argv, family,
                      [&](const std::string &name) { return run_hybrid(make(name)); });
}

} // namespace cora::bench

// ---------------------------------------  END OF CODE  ---------------------------------------- //
