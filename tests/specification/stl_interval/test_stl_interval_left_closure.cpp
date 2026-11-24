// test_stl_interval_left_closure - the interval with its lower end closed
//
// CORA: specification/@stlInterval/leftClosure.m
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


/// Closing the lower end.
void run(const std::string &b) {
    check(StlInterval(1, 2, false, false).left_closure() == StlInterval(1, 2, true, false),
          b + ": open");
    check(StlInterval().left_closure().is_empty_object(), b + ": empty");
    // the same with the checks off
    {
        ChecksScope off(false);
        check(StlInterval(1, 2, false).left_closure().left_closed(), b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_interval_left_closure");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
