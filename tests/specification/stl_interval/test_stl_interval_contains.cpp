// test_stl_interval_contains - whether an interval contains a time or an interval
//
// CORA: specification/@stlInterval/contains_.m
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


/// Numbers, intervals and the empty interval.
void run(const std::string &b) {
    const StlInterval I(1, 3, false, true);
    check(I.contains(2.0) && I.contains(3.0) && !I.contains(1.0) && !I.contains(4.0),
          b + ": numbers");
    check(I.contains(1.0 + 1e-12) && !StlInterval().contains(1.0), b + ": tolerance, empty");
    check(I.contains(StlInterval(1.5, 3)) && !I.contains(StlInterval(1, 2)) &&
              I.contains(StlInterval(1, 2, false, true)),
          b + ": an interval");
    check(I.contains(StlInterval()) && StlInterval().contains(StlInterval()) &&
              !StlInterval().contains(I),
          b + ": the empty interval");
    check(test::raises([&] { I.contains(1.0, -1.0); }, Error::wrong_value), b + ": negative tol");
    // the same with the checks off
    {
        ChecksScope off(false);
        check(StlInterval(0, 2).contains(1.0), b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_interval_contains");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
