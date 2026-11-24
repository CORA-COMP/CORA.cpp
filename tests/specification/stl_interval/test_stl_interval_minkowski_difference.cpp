// test_stl_interval_minkowski_difference - the Minkowski difference of time intervals
//
// CORA: specification/@stlInterval/minkDiff.m
// PROPERTY: specification/@stlInterval/minkDiff.m
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


/// Hand-computed cases, the unbounded cases, and the containment on a grid of times.
void run(const std::string &b) {
    check(StlInterval(1, 5).minkowski_difference(StlInterval(0, 2)) == StlInterval(1, 3),
          b + ": exact");
    check(StlInterval(1, 5).minkowski_difference(StlInterval(0, 2), "inner") == StlInterval(1, 3),
          b + ": inner");
    check(StlInterval(1, 5).minkowski_difference(StlInterval(2, 3)) == StlInterval(0, 2),
          b + ": clamps");
    check(StlInterval(1, 5).minkowski_difference(StlInterval()) ==
              StlInterval(0, stl_test::kInf, true, false),
          b + ": empty subtrahend");
    check(StlInterval().minkowski_difference(StlInterval(1)).is_empty_object(), b + ": empty");
    check(StlInterval(1, 5).minkowski_difference(StlInterval(0, stl_test::kInf)).is_empty_object(),
          b + ": by unbounded");
    // CORA gets NaN here and loses the unbounded end
    check(StlInterval(0, stl_test::kInf).minkowski_difference(StlInterval(0, stl_test::kInf)) ==
              StlInterval(0, stl_test::kInf),
          b + ": unbounded keeps its end");
    check(test::raises([] { StlInterval(1, 2).minkowski_difference(StlInterval(1), "x"); },
                       Error::wrong_value),
          b + ": type");
    Rng rng(4);
    const std::vector<StlInterval> samples = stl_test::samples(rng);
    // t + r lies in I for t in the difference and r in J
    for (const StlInterval &I : samples)
        for (const StlInterval &J : samples)
            for (const double t : stl_test::grid())
                for (const double r : stl_test::grid())
                    if (stl_test::member(I.minkowski_difference(J), t) && stl_test::member(J, r))
                        check(stl_test::member(I, t + r), b + ": minkDiff is inside");
    // the same with the checks off
    {
        ChecksScope off(false);
        check(StlInterval(1, 5).minkowski_difference(StlInterval(0, 2)) == StlInterval(1, 3),
              b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_interval_minkowski_difference");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
