// test_stl_interval_empty - the empty interval
//
// CORA: specification/@stlInterval/empty.m
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


/// The empty interval, and the refused dimension.
void run(const std::string &b) {
    check(StlInterval::empty().is_empty_object() && StlInterval::empty(1).dim() == 1,
          b + ": empty");
    check(test::raises([] { StlInterval::empty(2); }, Error::wrong_value), b + ": empty(2)");
    // the same with the checks off
    {
        ChecksScope off(false);
        check(StlInterval::empty().is_empty_object(), b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_interval_empty");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
