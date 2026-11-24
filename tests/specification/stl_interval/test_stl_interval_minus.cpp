// test_stl_interval_minus - the difference of a time interval and another interval
//
// CORA: specification/@stlInterval/minus.m
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


/// Hand-computed cases and the clamping of the lower bound at 0.
void run(const std::string &b) {
    const Interval J(Tensor::from_data({1.0}, {1}), Tensor::from_data({2.0}, {1}));
    check(StlInterval(1, 4, false, true).minus(J) == StlInterval(0, 3, true, true),
          b + ": Interval clamps");
    check(StlInterval(2, 5, true, false).minus(StlInterval(1, 2, true, false)) ==
              StlInterval(0, 4, false, false),
          b + ": minus");
    check(StlInterval(2, 5, true, true).minus(StlInterval(0, 1, false, true)) ==
              StlInterval(1, 5, true, false),
          b + ": ends");
    check(StlInterval(2, 5).minus(0.5) == StlInterval(1.5, 4.5), b + ": number");
    check(StlInterval(2, 5).minus(3.0) == StlInterval(0, 2), b + ": number clamps");
    check(StlInterval(2, 5).minus(StlInterval()).is_empty_object(), b + ": empty");
    // an invalid input is refused
    check(test::raises(
              [] {
                  StlInterval(1, 2).minus(Interval(Tensor::from_data({1.0, 2.0}, {2}),
                                                   Tensor::from_data({2.0, 3.0}, {2})));
              },
              Error::dimension_mismatch),
          b + ": a two-dimensional Interval");
    // the same with the checks off
    {
        ChecksScope off(false);
        check(StlInterval(2, 5).minus(0.5) == StlInterval(1.5, 4.5), b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_interval_minus");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
