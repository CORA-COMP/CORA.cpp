// test_interval_dirac - interval dirac: the Dirac delta and its derivatives on a box
//
// CORA: contSet/@interval/dirac.m
// CORA authors: Mark Wetzlinger, Adrian Kulmburg
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "testing.h"

#include <limits>

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

/// Whether the tensor holds exactly these numbers (infinities included).
bool aux_same(const Tensor &t, const std::vector<double> &values) { return t.data() == values; }

/// Six boxes: left of 0, right of 0, around 0, 0 as the lower end, 0 as the upper end, the point 0.
Interval aux_boxes() {
    return {column({-3.0, 2.0, -1.0, 0.0, -2.0, 0.0}), column({-1.0, 4.0, 1.0, 3.0, 0.0, 0.0})};
}

void dirac_of_order_0(const std::string &b) {
    const Interval R = aux_boxes().dirac();
    check(aux_same(R.inf(), std::vector<double>{0, 0, 0, 0, 0, 0}) &&
              aux_same(R.sup(), std::vector<double>{0, 0, kInf, kInf, kInf, kInf}),
          b + ": [0, 0] off 0, [0, Inf] on it");
    check(aux_same(aux_boxes().dirac(0).sup(), R.sup().data()), b + ": the default order is 0");
}

void dirac_of_order_1(const std::string &b) {
    const Interval R = aux_boxes().dirac(1);
    check(aux_same(R.inf(), std::vector<double>{0, 0, -kInf, -kInf, 0, -kInf}) &&
              aux_same(R.sup(), std::vector<double>{0, 0, kInf, 0, kInf, kInf}),
          b + ": [-Inf, 0] if 0 is the lower end, [0, Inf] if the upper end, else the line");
}

void dirac_of_higher_order(const std::string &b) {
    for (const int64_t n : {2, 5}) {
        const Interval R = aux_boxes().dirac(n);
        // orders of the Dirac box
        check(aux_same(R.inf(), std::vector<double>{0, 0, -kInf, -kInf, -kInf, -kInf}) &&
                  aux_same(R.sup(), std::vector<double>{0, 0, kInf, kInf, kInf, kInf}),
              b + ": order " + std::to_string(n) + " is the line on 0");
    }
    const Interval S = Interval::stack({aux_boxes(), aux_boxes()}).dirac(1);
    check(S.inf().shape() == std::vector<int64_t>({2, 6, 1}), b + ": a batch has a batch shape");
}

/// A negative order is refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval I = aux_boxes();
    check(test::raises([&] { I.dirac(-1); }, Error::wrong_value), "dirac: order -1");
    const Tensor on = I.dirac(1).sup();
    ChecksScope off(false);
    check(aux_same(I.dirac(1).sup(), on.data()), "dirac: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        dirac_of_order_0(b);
        dirac_of_order_1(b);
        dirac_of_higher_order(b);
    });
    checks_and_their_switch();
    return test::finish("interval dirac");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
