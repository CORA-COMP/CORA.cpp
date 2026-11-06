// test_interval_min - interval min: the elementwise smaller of two boxes
//
// CORA: contSet/@interval/min.m
// CORA authors: Tobias Ladner
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void min_takes_the_smaller_bounds(const std::string &b) {
    const Interval I(column({0.0, -1.0}), column({2.0, 1.0})),
        J(column({1.0, -2.0}), column({3.0, 0.0}));
    const Interval R = I.min(J);
    check(close(R.inf(), std::vector<double>{0, -2}) && close(R.sup(), std::vector<double>{2, 0}),
          b + ": [min(inf, inf2), min(sup, sup2)]");
    const Interval Q = I.min(column({1.0, 0.5}));
    check(close(Q.inf(), std::vector<double>{0, -1}) && close(Q.sup(), std::vector<double>{1, 0.5}),
          b + ": a column of numbers");
    check(close(I.min(Tensor::from_data({0.5}, {1, 1})).inf(), std::vector<double>{0, -1}),
          b + ": one number for every coordinate");
    const Zonotope Z(column({3.0, 3.0}), Tensor::from_data({1, 0, 0, 1}, {2, 2}));
    check(close(I.min(Z).inf(), std::vector<double>{0, -1}) &&
              close(I.min(Z).sup(), std::vector<double>{2, 1}),
          b + ": another set by its interval hull");
    check(close(I.min(), I.inf()), b + ": without an argument the lower bound");
    const Interval S = Interval::stack({I, J}).min(J);
    // a batch
    check(S.inf().shape() == std::vector<int64_t>({2, 2, 1}) &&
              close(S.sup().slice(0, 0, 1).reshape({2, 1}), R.sup()),
          b + ": a batch");
}

/// A set or an array of the wrong dimension is refused; the checks off give the same.
void checks_and_their_switch() {
    const Interval I(column({0.0, -1.0}), column({2.0, 1.0}));
    check(test::raises([&] { I.min(Interval(column({0.0}), column({1.0}))); },
                       Error::dimension_mismatch),
          "min: a box of dimension 1");
    check(test::raises([&] { I.min(column({1.0, 2.0, 3.0})); }, Error::dimension_mismatch),
          "min: three numbers for two coordinates");
    const Tensor on = I.min(column({1.0, 0.5})).sup();
    ChecksScope off(false);
    check(close(I.min(column({1.0, 0.5})).sup(), on, 0), "min: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(min_takes_the_smaller_bounds);
    checks_and_their_switch();
    return test::finish("interval min");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
