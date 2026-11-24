// test_stl_interval_plus - the sum of a time interval and another interval
//
// CORA: specification/@stlInterval/plus.m
// PROPERTY: specification/@stlInterval/plus.m
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


/// Hand-computed cases, and the enclosure of sums on a grid of times.
void run(const std::string &b) {
    check(StlInterval(1, 2, false, true).plus(StlInterval(2, 3)) == StlInterval(3, 5, false, true),
          b + ": plus");
    check(StlInterval(1, 2).plus(1.5) == StlInterval(2.5, 3.5), b + ": number");
    check(StlInterval(1, 2).plus(StlInterval()).is_empty_object(), b + ": empty");
    check(test::raises([] { StlInterval(1, 2).plus(-2.0); }, Error::wrong_input_in_constructor),
          b + ": negative");
    const Interval J(Tensor::from_data({1.0}, {1}), Tensor::from_data({2.0}, {1}));
    check(StlInterval(1, 4, false, true).plus(J) == StlInterval(2, 6, false, true),
          b + ": Interval");
    Rng rng(4);
    const std::vector<StlInterval> samples = stl_test::samples(rng);
    // s + r lies in the sum for s in I and r in J
    for (const StlInterval &I : samples)
        for (const StlInterval &S : samples)
            for (const double s : stl_test::grid())
                for (const double r : stl_test::grid())
                    if (stl_test::member(I, s) && stl_test::member(S, r))
                        check(stl_test::member(I.plus(S), s + r), b + ": plus encloses");
    // the same with the checks off
    {
        ChecksScope off(false);
        check(StlInterval(1, 2).plus(1.5) == StlInterval(2.5, 3.5), b + ": checks off");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(run);
    return test::finish("test_stl_interval_plus");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
