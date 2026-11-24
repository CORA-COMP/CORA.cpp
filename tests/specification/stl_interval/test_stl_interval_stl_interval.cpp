// test_stl_interval_stl_interval - the constructor forms and normalizations
//
// CORA: specification/@stlInterval/stlInterval.m
// CORA: specification/@stlInterval/dim.m
// CORA: specification/@stlInterval/isemptyobject.m
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


/// The forms of the constructor, the normalizations and the errors.
void run(const std::string &b) {
    // a point, open and half open ends, the empty interval by order, an infinite bound open
    const StlInterval p(1);
    check(p.lower() == 1 && p.upper() == 1 && p.left_closed() && p.right_closed(), b + ": point");
    check(!StlInterval(0, 1, false).left_closed() && !StlInterval(0, 1, false).right_closed(),
          b + ": open");
    check(StlInterval(0, 1, true, false).left_closed() &&
              !StlInterval(0, 1, true, false).right_closed(),
          b + ": half open");
    check(StlInterval(2, 1).is_empty_object(), b + ": lb > ub is empty");
    check(StlInterval(1, 1, false, true).is_empty_object(), b + ": an open point is empty");
    check(!StlInterval(0, stl_test::kInf).right_closed(), b + ": an infinite bound is open");
    check(StlInterval(stl_test::kInf, stl_test::kInf).is_empty_object(),
          b + ": infinite lower bound");
    check(StlInterval().dim() == 1 && StlInterval().is_empty_object(), b + ": dim and empty");
    // the errors of CORA's constructor
    check(test::raises([] { StlInterval(-1, 1); }, Error::wrong_input_in_constructor),
          b + ": negative");
    check(test::raises([] { StlInterval(std::nan(""), 1); }, Error::wrong_input_in_constructor),
          b + ": NaN");
    // the same with the checks off
    {
        ChecksScope off(false);
        check(StlInterval(1, 2).upper() == 2, b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_interval_stl_interval");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
