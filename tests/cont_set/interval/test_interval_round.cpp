// test_interval_round - interval round: the bounds of a box rounded to decimals
//
// CORA: contSet/@interval/round.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
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

void round_to_integers(const std::string &b) {
    // halves go away from zero
    const Interval I(column({-2.5, -0.4, 0.5, 1.49}), column({-1.5, 0.5, 2.5, 1.51}));
    const Interval R = I.round();
    check(R.inf().data() == std::vector<double>({-3, 0, 1, 1}) &&
              R.sup().data() == std::vector<double>({-2, 1, 3, 2}),
          b + ": the bounds to the nearest integer, halves away from zero");
    check(I.round(0, "decimals").inf().data() == R.inf().data(), b + ": the default is 0 decimals");
    const Interval S = Interval::stack({I, I}).round();
    check(S.inf().shape() == std::vector<int64_t>({2, 4, 1}), b + ": a batch has a batch shape");
    check(S.sup().slice(0, 1, 2).reshape({4, 1}).data() == R.sup().data(), b + ": a batch member");
}

void round_to_decimals(const std::string &b) {
    const Interval I(column({1.234, -1.236}), column({2.347, -0.004}));
    const Interval R = I.round(2);
    check(close(R.inf(), std::vector<double>{1.23, -1.24}) &&
              close(R.sup(), std::vector<double>{2.35, 0.0}),
          b + ": two decimals");
}

/// A negative number of digits, an unknown type and a type of MATLAB's round that is not
/// supported are refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval I(column({1.4}), column({2.6}));
    check(test::raises([&] { I.round(-1); }, Error::wrong_value), "round: negative digits");
    check(test::raises([&] { I.round(1, "digits"); }, Error::wrong_value), "round: unknown type");
    check(test::raises([&] { I.round(1, "significant"); }, Error::not_supported),
          "round: significant digits");
    const std::vector<double> on = I.round().sup().data();
    ChecksScope off(false);
    check(I.round().sup().data() == on, "round: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        round_to_integers(b);
        round_to_decimals(b);
    });
    checks_and_their_switch();
    return test::finish("interval round");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
