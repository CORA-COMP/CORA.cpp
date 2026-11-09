// benchmark_run - what every benchmark program shares: the instances to run and the MATLAB line
//
// A program names its family and how one instance runs; run_family runs the instances given on
// the command line (all of the family without any) and counts the verdicts that differ from
// MATLAB CORA's. An instance that fails (an unknown name, missing data) prints its error and
// counts as differing, so the program exits non-zero instead of aborting.
//
// Syntax:   return cora::bench::run_family(argc, argv, {"A01", "A02"}, run_one);
//           cora::bench::print_line("Platoon", "PLAA01-BND42", verified, seconds);
// See also: benchmark_data.h, benchmark_hybrid.h
//
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::bench {

/// The line 'benchmark,instance,result,time' of the MATLAB scripts.
inline void print_line(const std::string &benchmark, const std::string &instance, bool verified,
                       double seconds) {
    std::cout << benchmark << "," << instance << "," << verified << "," << seconds << '\n'
              << std::flush;
}

/// Runs `run` (true: the verdict is MATLAB's) on the named instances, else on the family; the
/// exit code is the number of instances that differ or fail.
inline int run_family(int argc, char **argv, const std::vector<std::string> &family,
                      const std::function<bool(const std::string &)> &run) {
    const std::vector<std::string> names =
        argc < 2 ? family : std::vector<std::string>(argv + 1, argv + argc);
    int differ = 0;
    for (const std::string &name : names) {
        try {
            if (!run(name)) ++differ;
        } catch (const std::exception &e) {
            std::cerr << name << ": " << e.what() << '\n';
            ++differ;
        }
    }
    return differ;
}

} // namespace cora::bench

// ---------------------------------------  END OF CODE  ---------------------------------------- //
