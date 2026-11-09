// benchmark_linear_verify_fast_arch23_iss - Spacestation benchmark of ARCH 2023
//
// The instances of MATLAB CORA's benchmark_linear_verifyFast_ARCH23_iss_<instance>.m, read from
// benchmarks/data and verified with LinearSys::verify.
//
// Syntax:   build/benchmarks/benchmark_linear_verifyFast_ARCH23_iss [instance ...]
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

    const std::vector<std::string> family = {"ISSC01_ISS02", "ISSC01_ISU02", "ISSF01_ISS01",
                                             "ISSF01_ISU01"};

    // Verification ----------------------------------------------------------------------------

    // the data of each instance; the exit code counts the verdicts that differ from MATLAB's
    int differ = 0;
    for (const std::string &instance : cora::bench::instance_names(argc, argv, family))
        if (!cora::bench::run_instance(instance)) ++differ;
    return differ;
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
