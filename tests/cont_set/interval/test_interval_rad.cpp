// test_interval_rad - interval rad: the half-width of each coordinate of a box
//
// CORA: contSet/@interval/rad.m

#include "cora/cont_set/interval/interval.h"
#include "testing.h"

#include <limits>

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

void rad_is_half_the_width(const std::string &b) {
    const Interval I(column({1.0, -2.0}), column({3.0, 2.0}));
    check(close(I.rad(), std::vector<double>{1.0, 2.0}), b + ": (sup - inf) / 2");
    const Interval M(Tensor::from_data({0, 1, 2, 3}, {2, 2}),
                     Tensor::from_data({2, 1, 2, 7}, {2, 2}));
    check(close(M.rad(), Tensor::from_data({1, 0, 0, 2}, {2, 2})), b + ": an interval matrix");
    const Interval U(column({0.0}), column({std::numeric_limits<double>::infinity()}));
    check(U.rad().data()[0] == std::numeric_limits<double>::infinity(), b + ": unbounded");
    const Tensor R = Interval::stack({I, I}).rad();
    check(R.shape() == std::vector<int64_t>({2, 2, 1}), b + ": a batch has a batch shape");
    check(close(R.slice(0, 1, 2).reshape({2, 1}), I.rad()), b + ": a batch member");
}

/// rad has no input to refuse; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval I(column({1.0, -2.0}), column({3.0, 2.0}));
    ChecksScope off(false);
    check(close(I.rad(), std::vector<double>{1.0, 2.0}), "rad: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(rad_is_half_the_width);
    checks_and_their_switch();
    return test::finish("interval rad");
}
