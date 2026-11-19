// test_interval_operators - the operators of an interval, as CORA writes them
//
// CORA: contSet/@interval/mtimes.m
// CORA: contSet/@interval/plus.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

Interval box() { return Interval(test::vec({-1.0, 0.0}), test::vec({1.0, 2.0})); }

void sums_and_translations(const std::string &b) {
    const Interval S = box() + box();
    check(close(S.inf(), test::vec({-2.0, 0.0})) && close(S.sup(), test::vec({2.0, 4.0})),
          b + ": I + I2");
    const Tensor v = test::vec({1.0, 1.0});
    check(close((box() + v).inf(), test::vec({0.0, 1.0})) &&
              close((v + box()).sup(), test::vec({2.0, 3.0})),
          b + ": I + v translates");
    check(close((box() - v).sup(), test::vec({0.0, 1.0})), b + ": I - v");
}

// a number scales both bounds; -1 swaps them
void numbers_scale_and_a_negative_one_swaps_the_bounds(const std::string &b) {
    const Interval J = 2.0 * box();
    check(close(J.inf(), test::vec({-2.0, 0.0})) && close(J.sup(), test::vec({2.0, 4.0})),
          b + ": s * I");
    const Interval N = -box();
    check(close(N.inf(), test::vec({-1.0, -2.0})) && close(N.sup(), test::vec({1.0, 0.0})),
          b + ": -I");
}

void a_matrix_maps_the_box(const std::string &b) {
    const Tensor M({{0.0, 1.0}, {1.0, 0.0}});
    // M * I is the hull of the image
    const Interval J = M * box();
    check(close(J.inf(), test::vec({0.0, -1.0})) && close(J.sup(), test::vec({2.0, 1.0})),
          b + ": M * I is the hull of the image");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        sums_and_translations(b);
        numbers_scale_and_a_negative_one_swaps_the_bounds(b);
        a_matrix_maps_the_box(b);
    });
    return test::finish("interval operators");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
