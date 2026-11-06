// test_interval_plus - interval plus: the Minkowski sum
//
// CORA: contSet/@interval/plus.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void plus_is_the_minkowski_sum(const std::string &b) {
    const Interval A(column({0.0, 1.0}), column({1.0, 3.0})),
        B(column({-2.0, 0.5}), column({0.0, 0.5}));
    const Interval S = A.plus(B);
    check(close(S.inf(), std::vector<double>{-2.0, 1.5}) &&
              close(S.sup(), std::vector<double>{1.0, 3.5}),
          b + ": bounds add");
}

/// Sets of different dimensions are refused; with the checks off the sum is the same.
void checks_and_their_switch() {
    const Interval S = Interval(test::column({0, -1}), test::column({1, 1})),
                   S3 = Interval(test::column({0, -1, 0}), test::column({1, 1, 2}));
    check(test::raises([&] { S.plus(S3); }, Error::dimension_mismatch), "plus: dimensions 2 and 3");
    const Tensor on = S.plus(S).center();
    ChecksScope off(false);
    check(test::close(S.plus(S).center(), on, 0), "plus: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { plus_is_the_minkowski_sum(b); });
    checks_and_their_switch();
    return test::finish("interval plus");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
