// benchmark_linear_verifyFast_ARCH23_heat3D - heat3D benchmark of ARCH 2023
//
// The instances of MATLAB CORA's benchmark_linear_verifyFast_ARCH23_heat3D_<instance>.m, read from
// benchmarks/data and verified with LinearSys::verify.
//
// Syntax:   build/benchmarks/benchmark_linear_verifyFast_ARCH23_heat3D [instance ...]
// Outputs:  one line 'benchmark,instance,result,time' per instance, as the MATLAB scripts
// See also: benchmarkData.h, the Python script of the same name

#include "benchmarkData.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

int main(int argc, char **argv) {
    // Parameters ------------------------------------------------------------------------------

    const std::vector<std::string> family = {
        "HEAT01", "HEAT02"};

    // Verification ----------------------------------------------------------------------------

    // system, R0, U, specifications and verifyAlg come from the data of each instance
    for (const std::string &instance : cora::bench::instanceNames(argc, argv, family))
        cora::bench::runInstance(instance);

    return 0;
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
