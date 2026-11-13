// test_benchmark_run - the switches of the benchmark programs: on, off, unset
//
// env_flag reads a variable such as CORACPP_BENCHMARK_VERBOSE by its value: unset, empty and "0"
// are off, anything else is on. (Read as "set", CORACPP_BENCHMARK_VERBOSE=0 turned the details of
// Double Pendulum less-robust on and changed its time step.)
//
// Authors:       Claude Sonnet 5.5
// Written:       13-November-2026
// Last update:   ---
// Reviewed:      ---

#include "benchmark_run.h"
#include "testing.h"

#include <cstdlib>

using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    const char *name = "CORACPP_TEST_SWITCH";
    unsetenv(name);
    check(!cora::bench::env_flag(name), "unset is off");
    setenv(name, "", 1);
    check(!cora::bench::env_flag(name), "empty is off");
    setenv(name, "0", 1);
    check(!cora::bench::env_flag(name), "0 is off");
    for (const char *on : {"1", "yes", "true", "00"}) {
        setenv(name, on, 1);
        check(cora::bench::env_flag(name), std::string(on) + " is on");
    }
    unsetenv(name);
    return test::finish("benchmark_run");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
