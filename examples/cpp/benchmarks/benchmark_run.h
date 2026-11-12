// benchmark_run - what every benchmark program shares: the instances to run and the MATLAB line
//
// A program names its family and how one instance runs; run_family runs the instances given on
// the command line (all of the family without any) and counts the verdicts that differ from
// MATLAB CORA's. An instance that fails (an unknown name, missing data) prints its error and
// counts as differing, so the program exits non-zero instead of aborting. An instance that needs
// more memory than the machine has (require_memory_gb) is skipped: exit code 77 if all were.
// linear_step() reads the linear step of the nonlinear algorithms from CORACPP_LINEAR_STEP (cora,
// closed_form, coralean) for the programs that run a lin or lin-adaptive instance.
//
// Syntax:   return cora::bench::run_family(argc, argv, {"A01", "A02"}, run_one);
//           cora::bench::require_memory_gb(24);   // in run_one, before the work
//           cora::bench::print_line("Platoon", "PLAA01-BND42", verified, seconds);
// See also: benchmark_data.h, benchmark_hybrid.h
//
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_dynamics/algorithms.h"

#include <cstdlib>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
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

/// The linear step of the nonlinear algorithms: CORACPP_LINEAR_STEP = closed_form (default), cora
/// or coralean.
inline cora::alg::LinearStep linear_step() {
    const char *value = std::getenv("CORACPP_LINEAR_STEP");
    const std::string name = value ? value : "closed_form";
    if (name == "cora") return cora::alg::LinearStep::cora;
    if (name == "coralean") return cora::alg::LinearStep::coralean;
    if (name != "closed_form")
        throw std::runtime_error("CORACPP_LINEAR_STEP must be cora, closed_form or coralean");
    return cora::alg::LinearStep::closed_form;
}

/// Thrown by require_memory_gb: the instance is skipped, not failed.
struct NotEnoughMemory : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// Skips the instance unless `gb` GB of memory are available (Linux: MemAvailable; elsewhere the
/// check passes). The sets of a long reach run are all kept, which a hosted runner (16 GB) cannot.
inline void require_memory_gb(double gb) {
    std::ifstream meminfo("/proc/meminfo");
    std::string key, unit;
    double kb = 0;
    while (meminfo >> key >> kb >> unit)
        if (key == "MemAvailable:") {
            if (kb / 1048576.0 < gb)
                throw NotEnoughMemory("skipped: needs " + std::to_string(int(gb)) + " GB, " +
                                      std::to_string(int(kb / 1048576.0)) + " GB available");
            return;
        }
}

/// Runs `run` (true: the verdict is MATLAB's) on the named instances, else on the family; the
/// exit code is the number of instances that differ or fail, 77 if all were skipped.
inline int run_family(int argc, char **argv, const std::vector<std::string> &family,
                      const std::function<bool(const std::string &)> &run) {
    const std::vector<std::string> names =
        argc < 2 ? family : std::vector<std::string>(argv + 1, argv + argc);
    int differ = 0;
    std::size_t skipped = 0;
    for (const std::string &name : names) {
        try {
            if (!run(name)) ++differ;
        } catch (const NotEnoughMemory &e) {
            std::cerr << name << ": " << e.what() << '\n';
            ++skipped;
        } catch (const std::exception &e) {
            std::cerr << name << ": " << e.what() << '\n';
            ++differ;
        }
    }
    return differ == 0 && skipped == names.size() ? 77 : differ;
}

} // namespace cora::bench

// ---------------------------------------  END OF CODE  ---------------------------------------- //
