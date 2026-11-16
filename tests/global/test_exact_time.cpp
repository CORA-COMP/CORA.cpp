// test_exact_time - the exact sum of doubles that the adaptive algorithms keep as their time
//
// Sums that doubles round are exact (0.1 ten times is not 1, and 1 + 2^-60 - 1 is 2^-60), the
// comparisons are exact, nearest rounds ties to even, and rat_up is CORALean's ratUp at nearest:
// a numerator of more than 53 bits is rounded as it is built from its highest bit.
//
// Authors:       Claude Opus 5.5
// Written:       16-November-2026
// Last update:   ---
// Reviewed:      ---

#include "global/exact_time.h"
#include "testing.h"

#include <cmath>

using cora::ExactTime;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    // exact sums and differences, and the comparisons on them
    ExactTime t;
    for (int k = 0; k < 10; ++k) t = t + 0.1;
    check(!(t <= ExactTime(1.0)) || !(ExactTime(1.0) <= t), "ten times 0.1 is not 1");
    check(t.nearest() == 1.0, "ten times 0.1 rounds to 1");
    const ExactTime tiny = ExactTime(1.0) + std::ldexp(1.0, -60) - ExactTime(1.0);
    check(tiny.nearest() == std::ldexp(1.0, -60), "1 + 2^-60 - 1 is 2^-60");
    check(ExactTime(0.0) < tiny && ExactTime(-1.0) < ExactTime(0.0), "the order");
    check((ExactTime(0.25) - ExactTime(0.5)).nearest() == -0.25, "a negative difference");
    check(ExactTime(0.5).times(3).nearest() == 1.5, "three times 0.5");
    const double least = std::ldexp(1.0, -1074);
    check(ExactTime(least).nearest() == least && ExactTime(1e300).nearest() == 1e300,
          "the least and a large double");

    // nearest: a tie rounds to the even mantissa
    const double ulp = std::ldexp(1.0, -52);
    check((ExactTime(1.0) + ulp / 2).nearest() == 1.0, "1 + ulp/2 ties to 1");
    check((ExactTime(1.0 + ulp) + ulp / 2).nearest() == 1.0 + 2 * ulp, "a tie to even upward");
    check((ExactTime(1.0) + ulp / 2 + ulp / 1024).nearest() == 1.0 + ulp, "above the tie");

    // rat_up: exact below 2^53 bits of the numerator, rounded as CORALean builds it above
    check(ExactTime(0.015).rat_up() == 0.015 && (ExactTime(1.0) + 0.5).rat_up() == 1.5,
          "a double is its own ratUp");
    check((ExactTime(-1.0)).rat_up() == 0.0, "ratUp of a negative number is 0");
    // 2^54 + 3: the bits 1 0 ... 0 1 1; v = 2^53 + 1 ties to 2^53, then 2^54 + 1 rounds to 2^54
    const ExactTime odd = ExactTime(std::ldexp(1.0, 54)) + 3.0;
    check(odd.nearest() == std::ldexp(1.0, 54) + 4 && odd.rat_up() == std::ldexp(1.0, 54),
          "ratUp rounds the numerator bit by bit, nearest at once");
    // the oracle's text: the integer of 2^-1074 in decimal
    check(ExactTime(std::ldexp(1.0, -1074)).text() == "1*2^-1074", "the text of the least double");
    check(ExactTime(-std::ldexp(1.0, -1073)).text() == "-2*2^-1074", "a negative text");
    check(ExactTime().text() == "0*2^-1074", "the text of 0");
    const std::string one = ExactTime(1.0).text(); // 2^1074 has 324 digits
    check(one.size() == 324 + 8 && one.rfind("2024022533", 0) == 0, "the text of 1");
    return test::finish("exact_time");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
