// test_stl_interval_interval - the closed Interval of the bounds
//
// CORA: specification/@stlInterval/interval.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/settings.h"
#include "cora/specification/stl_interval.h"
#include "stl_interval_testing.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {


/// The Interval of the bounds, and of the empty interval.
void run(const std::string &b) {
    const Interval J = StlInterval(1, 2, false, false).interval();
    check(J.inf().item() == 1 && J.sup().item() == 2, b + ": interval");
    check(StlInterval().interval().is_empty(), b + ": empty");
    // the same with the checks off
    {
        ChecksScope off(false);
        check(StlInterval(1, 2).interval().inf().item() == 1, b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_interval_interval");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
