// benchmark_linear_verify_ARCH23_rand - random benchmark of ARCH 2023
//
// The instances of MATLAB CORA's benchmark_linear_verify_ARCH23_rand_<instance>.m, read from
// benchmarks/data and verified with LinearSys::verify.
//
// Syntax:   build/benchmarks/benchmark_linear_verify_ARCH23_rand [instance ...]
// Outputs:  one line 'benchmark,instance,result,time' per instance, as the MATLAB scripts
// See also: benchmark_data.h, the Python script of the same name

#include "benchmark_data.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

int main(int argc, char **argv) {
    // Parameters ------------------------------------------------------------------------------

    const std::vector<std::string> family = {
        "RAND01", "RAND02"};

    // Verification ----------------------------------------------------------------------------

    // system, R0, U, specifications and verifyAlg come from the data of each instance
    for (const std::string &instance : cora::bench::instance_names(argc, argv, family))
        cora::bench::run_instance(instance);

    return 0;
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
