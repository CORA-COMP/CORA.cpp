// test_stl_interval_supremum - the upper bound
//
// CORA: specification/@stlInterval/supremum.m
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


/// The supremum with its closedness; the empty interval has none.
void run(const std::string &b) {
    const StlEndpoint hi = StlInterval(1, 2, false, true).supremum();
    check(hi.value == 2 && hi.closed, b + ": supremum");
    check(std::isnan(StlInterval().supremum().value), b + ": empty");
    // the same with the checks off
    {
        ChecksScope off(false);
        check(StlInterval(1, 2).supremum().value == 2, b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_interval_supremum");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
