// test_interval_minus - interval minus: the difference of a box and a box or a point
//
// CORA: contSet/@interval/minus.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void minus_subtracts_the_opposite_bounds(const std::string &b) {
    const Interval I(column({0.0, 1.0}), column({2.0, 4.0})),
        J(column({-1.0, 0.0}), column({1.0, 3.0}));
    const Interval R = I.minus(J);
    check(close(R.inf(), std::vector<double>{-1, -2}) && close(R.sup(), std::vector<double>{3, 4}),
          b + ": [inf - sup2, sup - inf2]");
    const Interval O = I - J;
    check(close(O.inf(), R.inf()) && close(O.sup(), R.sup()), b + ": the operator");
    const Interval V = I.minus(column({1.0, 1.0}));
    check(close(V.inf(), std::vector<double>{-1, 0}) && close(V.sup(), std::vector<double>{1, 3}),
          b + ": a column shifts both bounds");
    check(close(I.minus(Tensor::from_data({1.0}, {1, 1})).inf(), std::vector<double>{-1, 0}),
          b + ": a number");
    const Interval W = Interval::minus(column({1.0, 1.0}), I);
    check(close(W.inf(), std::vector<double>{-1, -3}) && close(W.sup(), std::vector<double>{1, 0}),
          b + ": a point minus a box is [v - sup, v - inf]");
    const Interval S = Interval::stack({I, J}) - column({1.0, 1.0});
    // a batch
    check(S.inf().shape() == std::vector<int64_t>({2, 2, 1}) &&
              close(S.sup().slice(0, 1, 2).reshape({2, 1}), std::vector<double>{0, 2}),
          b + ": a batch");
}

/// A box of another dimension and an array of another shape are refused; the checks off give the
/// same.
void checks_and_their_switch() {
    const Interval I(column({0.0, 1.0}), column({2.0, 4.0}));
    check(test::raises([&] { I.minus(Interval(column({0.0}), column({1.0}))); },
                       Error::dimension_mismatch),
          "minus: a box of dimension 1");
    check(test::raises([&] { I.minus(column({1.0, 2.0, 3.0})); }, Error::dimension_mismatch),
          "minus: three numbers for two coordinates");
    check(test::raises([&] { Interval::minus(column({1.0, 2.0, 3.0}), I); },
                       Error::dimension_mismatch),
          "minus: a point of three coordinates minus a box");
    const Tensor on = I.minus(I).sup();
    ChecksScope off(false);
    check(close(I.minus(I).sup(), on, 0), "minus: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(minus_subtracts_the_opposite_bounds);
    checks_and_their_switch();
    return test::finish("interval minus");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
