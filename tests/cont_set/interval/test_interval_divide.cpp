// test_interval_divide - interval divide: a box divided by a number (CORA's mrdivide)
//
// CORA: contSet/@interval/mrdivide.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner, Claude Opus 5.5
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

constexpr double kInf = std::numeric_limits<double>::infinity();

void divide_scales_and_a_negative_number_swaps(const std::string &b) {
    const Interval I(vec({2.0, -4.0}), vec({4.0, 6.0}));
    const Interval R = I.divide(2.0);
    check(close(R.inf(), std::vector<double>{1, -2}) && close(R.sup(), std::vector<double>{2, 3}),
          b + ": a positive number");
    const Interval N = I / -2.0;
    check(close(N.inf(), std::vector<double>{-2, -3}) && close(N.sup(), std::vector<double>{-1, 2}),
          b + ": a negative number swaps the bounds (the operator)");
    const Interval Z = I.divide(0.0);
    check(Z.inf().data() == std::vector<double>({-kInf, -kInf}) &&
              Z.sup().data() == std::vector<double>({kInf, kInf}),
          b + ": 0 gives the whole line, as CORA");
    const Interval U = Interval(vec({1.0}), vec({kInf})).divide(2.0);
    check(U.sup().data()[0] == kInf && close(U.inf(), std::vector<double>{0.5}),
          b + ": an infinite bound stays");
    const Interval S = Interval::stack({I, I}).divide(2.0);
    // a batch
    check(S.inf().shape() == std::vector<int64_t>({2, 2}) &&
              close(S.sup().slice(0, 1, 2).reshape({2}), R.sup()),
          b + ": a batch");
}

/// An infinite bound divided by an infinity is NaN and refused; the checks off give the same.
void checks_and_their_switch() {
    const Interval I(vec({2.0, -4.0}), vec({4.0, 6.0}));
    check(
        test::raises([] { Interval(vec({1.0}), vec({kInf})).divide(kInf); }, Error::out_of_domain),
        "divide: Inf / Inf");
    const Tensor on = I.divide(2.0).sup();
    ChecksScope off(false);
    check(close(I.divide(2.0).sup(), on, 0), "divide: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(divide_scales_and_a_negative_number_swaps);
    checks_and_their_switch();
    return test::finish("interval divide");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
