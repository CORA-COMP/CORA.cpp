// test_stl_interval_infimum - the lower bound
//
// CORA: specification/@stlInterval/infimum.m
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


/// The infimum with its closedness; the empty interval has none.
void run(const std::string &b) {
    const StlEndpoint lo = StlInterval(1, 2, false, true).infimum();
    check(lo.value == 1 && !lo.closed, b + ": infimum");
    check(std::isnan(StlInterval().infimum().value), b + ": empty");
    // the same with the checks off
    {
        ChecksScope off(false);
        check(StlInterval(1, 2).infimum().value == 1, b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_interval_infimum");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
