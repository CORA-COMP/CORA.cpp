// test_stl_interval_to_left - the times before the interval
//
// CORA: specification/@stlInterval/toLeft.m
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


/// The times from 0 up to the start; the empty interval is an error.
void run(const std::string &b) {
    check(StlInterval(1, 2, true, true).to_left() == StlInterval(0, 1, true, false),
          b + ": to_left");
    check(StlInterval(0, 2, true, true).to_left().is_empty_object(), b + ": at 0");
    check(test::raises([] { StlInterval().to_left(); }, Error::empty_set), b + ": empty");
    // the same with the checks off
    {
        ChecksScope off(false);
        check(StlInterval(1, 2).to_left() == StlInterval(0, 1, true, false), b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_interval_to_left");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
