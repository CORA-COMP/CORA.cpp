// test_stl_interval_and_ - the intersection of two time intervals
//
// CORA: specification/@stlInterval/and_.m
// PROPERTY: specification/@stlInterval/and_.m
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


/// Hand-computed cases, and the set semantics on a grid of times.
void run(const std::string &b) {
    check(StlInterval(0, 2, true, false).and_(StlInterval(1, 3, false, true)) ==
              StlInterval(1, 2, false, false),
          b + ": and_");
    check(StlInterval(0, 2).and_(StlInterval(0, 2, false)) == StlInterval(0, 2, false),
          b + ": ties");
    check(StlInterval(0, 1).and_(StlInterval(2, 3)).is_empty_object(), b + ": disjoint");
    Rng rng(4);
    const std::vector<StlInterval> samples = stl_test::samples(rng);
    // a time is in the intersection exactly where it is in both
    for (const StlInterval &I : samples)
        for (const StlInterval &J : samples)
            for (const double t : stl_test::grid())
                check(stl_test::member(I.and_(J), t) ==
                          (stl_test::member(I, t) && stl_test::member(J, t)),
                      b + ": and_ is the intersection");
    // an invalid input is refused
    check(test::raises([] { StlInterval(-1, 1).and_(StlInterval(1)); },
                       Error::wrong_input_in_constructor),
          b + ": an invalid operand");
    // the same with the checks off
    {
        ChecksScope off(false);
        check(StlInterval(0, 2).and_(StlInterval(1, 3)) == StlInterval(1, 2), b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_interval_and_");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
