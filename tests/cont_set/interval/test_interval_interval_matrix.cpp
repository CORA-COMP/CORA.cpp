// test_interval_interval_matrix - interval interval_matrix: the box of its center and radius
//
// CORA: contSet/@interval/intervalMatrix.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

void interval_matrix_is_center_plus_minus_radius(const std::string &b) {
    const Tensor lo = Tensor::from_data({1, 2, 3, 4}, {2, 2});
    const Interval M(lo, lo + 2.0);
    const Interval R = M.interval_matrix();
    check(R.inf().shape() == std::vector<int64_t>({2, 2}) && close(R.inf(), lo, 0) &&
              close(R.sup(), lo + 2.0, 0),
          b + ": a matrix of exact bounds is itself");
    check(close(R.inf(), M.center() - M.rad(), 0) && close(R.sup(), M.center() + M.rad(), 0),
          b + ": c - r and c + r");
    const Interval S = Interval::stack({M, M}).interval_matrix();
    check(S.inf().shape() == std::vector<int64_t>({2, 2, 2}), b + ": a batch has a batch shape");
}

/// Equal to the interval up to the rounding of c - r and c + r.
void interval_matrix_is_the_interval_up_to_rounding(const std::string &b) {
    cora::Rng rng(35);
    for (int trial = 0; trial < 20; ++trial) {
        const Interval I = Interval::generate_random(rng, 4);
        const Interval R = I.interval_matrix();
        check(close(R.inf(), I.inf(), 1e-15) && close(R.sup(), I.sup(), 1e-15),
              b + ": equal up to rounding");
    }
}

/// interval_matrix has no input to refuse; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval I(column({-1.0, 0.0}), column({1.0, 1.0}));
    const Tensor on = I.interval_matrix().sup();
    ChecksScope off(false);
    check(close(I.interval_matrix().sup(), on, 0), "interval_matrix: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        interval_matrix_is_center_plus_minus_radius(b);
        interval_matrix_is_the_interval_up_to_rounding(b);
    });
    checks_and_their_switch();
    return test::finish("interval interval_matrix");
}
