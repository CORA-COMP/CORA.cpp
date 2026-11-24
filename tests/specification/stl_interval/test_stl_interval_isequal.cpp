// test_stl_interval_isequal - whether two time intervals are equal
//
// CORA: specification/@stlInterval/isequal.m
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


/// Equal up to tol, the ends, empty and unbounded intervals.
void run(const std::string &b) {
    const StlInterval I(1, 3, false, true);
    check(I.isequal(StlInterval(1, 3 + 1e-14, false, true)), b + ": tolerance");
    check(!I.isequal(StlInterval(1, 3)) && !I.isequal(StlInterval()), b + ": ends, empty");
    check(StlInterval().isequal(StlInterval()) &&
              StlInterval(0, stl_test::kInf).isequal(StlInterval(0, stl_test::kInf)),
          b + ": empty, unbounded");
    // an invalid input is refused
    check(test::raises([] { StlInterval(1).isequal(StlInterval(1), -1.0); }, Error::wrong_value),
          b + ": negative tol");
    // the same with the checks off
    {
        ChecksScope off(false);
        check(StlInterval(1, 2).isequal(StlInterval(1, 2)), b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_interval_isequal");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
