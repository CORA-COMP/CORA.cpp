// test_interval_enlarge - interval enlarge: a box scaled about its center
//
// CORA: contSet/@interval/enlarge.m
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "testing.h"

#include <cmath>
#include <limits>

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

void enlarge_scales_about_the_center(const std::string &b) {
    const Interval I(column({0.0, 1.0}), column({2.0, 5.0}));
    const Interval R = I.enlarge(2.0);
    check(close(R.inf(), std::vector<double>{-1, -1}) && close(R.sup(), std::vector<double>{3, 7}),
          b + ": a number scales every dimension");
    const Interval Q = I.enlarge(column({0.5, 0.0}));
    check(close(Q.inf(), std::vector<double>{0.5, 3}) &&
              close(Q.sup(), std::vector<double>{1.5, 3}),
          b + ": a column scales each dimension, 0 gives the center");
    check(close(I.enlarge(1.0).inf(), I.inf()) && close(I.enlarge(1.0).sup(), I.sup()),
          b + ": a factor of 1 keeps the box");
    const Interval S = Interval::stack({I, I}).enlarge(2.0);
    check(S.inf().shape() == std::vector<int64_t>({2, 2, 1}), b + ": a batch has a batch shape");
    check(close(S.sup().slice(0, 1, 2).reshape({2, 1}), R.sup()), b + ": a batch member");
}

/// An unbounded half-width grows to the whole line for f > 1; f < 1 is refused.
void enlarge_unbounded_boxes(const std::string &b) {
    const Interval U = Interval::unbounded(2);
    check(U.enlarge(2.0).inf().data() == std::vector<double>({-kInf, -kInf}) &&
              U.enlarge(2.0).sup().data() == std::vector<double>({kInf, kInf}),
          b + ": the whole space stays");
    check(U.enlarge(1.0).inf().data() == std::vector<double>({-kInf, -kInf}),
          b + ": f = 1 on the space");
    const Interval half_line(column({0.0, -kInf}), column({kInf, 3.0}));
    check(half_line.enlarge(1.0).inf().data() == std::vector<double>({0.0, -kInf}) &&
              half_line.enlarge(1.0).sup().data() == std::vector<double>({kInf, 3.0}),
          b + ": f = 1 keeps a half-line (CORA gives NaN)");
    const Interval H(column({0.0, 0.0}), column({kInf, 4.0}));
    const Interval R = H.enlarge(column({3.0, 0.5}));
    check(std::isinf(R.inf().data()[0]) && R.inf().data()[0] < 0 && std::isinf(R.sup().data()[0]),
          b + ": a half-line becomes the line");
    check(close(R.inf().slice(-2, 1, 2), std::vector<double>{1.0}) &&
              close(R.sup().slice(-2, 1, 2), std::vector<double>{3.0}),
          b + ": a bounded dimension scales");
}

/// A negative factor, a factor of the wrong shape, and f < 1 on an unbounded box are refused;
/// with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval I(column({0.0, 1.0}), column({2.0, 5.0}));
    check(test::raises([&] { I.enlarge(-1.0); }, Error::wrong_value), "enlarge: a negative factor");
    check(test::raises([&] { I.enlarge(column({1.0, 1.0, 1.0})); }, Error::dimension_mismatch),
          "enlarge: three factors for two dimensions");
    check(test::raises([&] { Interval::unbounded(1).enlarge(0.5); }, Error::not_supported),
          "enlarge: f < 1 on an unbounded box");
    const Tensor on = I.enlarge(2.0).sup();
    ChecksScope off(false);
    check(close(I.enlarge(2.0).sup(), on, 0), "enlarge: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        enlarge_scales_about_the_center(b);
        enlarge_unbounded_boxes(b);
    });
    checks_and_their_switch();
    return test::finish("interval enlarge");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
