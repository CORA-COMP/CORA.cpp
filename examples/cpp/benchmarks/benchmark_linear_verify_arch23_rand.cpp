// benchmark_linear_verify_arch23_rand - random benchmark of ARCH 2023
//
// The instances of MATLAB CORA's benchmark_linear_verify_ARCH23_rand_<instance>.m, read from
// benchmarks/data and verified with LinearSys::verify.
//
// Syntax:   build/benchmarks/benchmark_linear_verify_ARCH23_rand [instance ...]
// Outputs:  one line 'benchmark,instance,result,time' per instance, as the MATLAB scripts
// See also: benchmark_data.h, the Python script of the same name
//
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   09-November-2026
// Reviewed:      ---

#include "benchmark_data.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main(int argc, char **argv) {
    // Parameters ------------------------------------------------------------------------------

    const std::vector<std::string> family = {"RAND01", "RAND02"};

    // Verification ----------------------------------------------------------------------------

    // the data of each instance; the exit code counts the verdicts that differ from MATLAB's
    return cora::bench::run_family(argc, argv, family, cora::bench::run_instance);
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
