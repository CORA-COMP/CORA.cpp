// test_interval_half_width - interval half_width: the half-width of each coordinate of a box
//
// CORA: contSet/@interval/rad.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "testing.h"

#include <limits>

using namespace cora;
using test::check;
using test::close;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void rad_is_half_the_width(const std::string &b) {
    const Interval I(vec({1.0, -2.0}), vec({3.0, 2.0}));
    check(close(I.half_width(), std::vector<double>{1.0, 2.0}), b + ": (sup - inf) / 2");
    const Interval M = Interval::matrix(Tensor::from_data({0, 1, 2, 3}, {2, 2}),
                                        Tensor::from_data({2, 1, 2, 7}, {2, 2}));
    check(close(M.half_width(), Tensor::from_data({1, 0, 0, 2}, {2, 2})),
          b + ": an interval matrix");
    const Interval U(vec({0.0}), vec({std::numeric_limits<double>::infinity()}));
    check(U.half_width().data()[0] == std::numeric_limits<double>::infinity(), b + ": unbounded");
    const Tensor R = Interval::stack({I, I}).half_width();
    check(R.shape() == std::vector<int64_t>({2, 2}), b + ": a batch has a batch shape");
    check(close(R.slice(0, 1, 2).reshape({2}), I.half_width()), b + ": a batch member");
}

/// rad has no input to refuse; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval I(vec({1.0, -2.0}), vec({3.0, 2.0}));
    ChecksScope off(false);
    check(close(I.half_width(), std::vector<double>{1.0, 2.0}),
          "rad: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(rad_is_half_the_width);
    checks_and_their_switch();
    return test::finish("interval rad");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
