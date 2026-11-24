// test_stl_interval_to_right - the times after the interval
//
// CORA: specification/@stlInterval/toRight.m
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


/// The times after the end; the empty interval is an error.
void run(const std::string &b) {
    check(StlInterval(1, 2, true, true).to_right() == StlInterval(2, stl_test::kInf, false, false),
          b + ": to_right");
    check(StlInterval(1, stl_test::kInf).to_right().is_empty_object(), b + ": unbounded");
    check(test::raises([] { StlInterval().to_right(); }, Error::empty_set), b + ": empty");
    // the same with the checks off
    {
        ChecksScope off(false);
        check(StlInterval(1, 2).to_right() == StlInterval(2, stl_test::kInf, false, false),
              b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_interval_to_right");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
