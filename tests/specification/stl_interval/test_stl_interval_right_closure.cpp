// test_stl_interval_right_closure - the interval with its upper end closed
//
// CORA: specification/@stlInterval/rightClosure.m
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


/// Closing the upper end, not an infinite one.
void run(const std::string &b) {
    check(StlInterval(1, 2, false, false).right_closure() == StlInterval(1, 2, false, true),
          b + ": open");
    check(StlInterval(1, stl_test::kInf, false).right_closure() ==
              StlInterval(1, stl_test::kInf, false),
          b + ": unbounded");
    check(StlInterval().right_closure().is_empty_object(), b + ": empty");
    // the same with the checks off
    {
        ChecksScope off(false);
        check(StlInterval(1, 2, false).right_closure().right_closed(), b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_interval_right_closure");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
