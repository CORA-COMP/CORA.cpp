// test_interval_operators - the operators of an interval, as CORA writes them

#include "contSet/interval/interval.h"
#include "testing.h"

using namespace cora::ct;
using test::check;
using test::close;

namespace {

Interval box() { return Interval(Tensor({-1.0, 0.0}), Tensor({1.0, 2.0})); }

void sums_and_translations(const std::string &b) {
    const Interval S = box() + box();
    check(close(S.inf, Tensor({-2.0, 0.0})) && close(S.sup, Tensor({2.0, 4.0})), b + ": I + I2");
    const Tensor v({1.0, 1.0});
    check(close((box() + v).inf, Tensor({0.0, 1.0})) && close((v + box()).sup, Tensor({2.0, 3.0})),
          b + ": I + v translates");
    check(close((box() - v).sup, Tensor({0.0, 1.0})), b + ": I - v");
}

void numbers_scale_and_a_negative_one_swaps_the_bounds(const std::string &b) {
    const Interval J = 2.0 * box();
    check(close(J.inf, Tensor({-2.0, 0.0})) && close(J.sup, Tensor({2.0, 4.0})), b + ": s * I");
    const Interval N = -box();
    check(close(N.inf, Tensor({-1.0, -2.0})) && close(N.sup, Tensor({1.0, 0.0})), b + ": -I");
}

void a_matrix_maps_the_box(const std::string &b) {
    const Tensor M({{0.0, 1.0}, {1.0, 0.0}});
    const Interval J = M * box();
    check(close(J.inf, Tensor({0.0, -1.0})) && close(J.sup, Tensor({2.0, 1.0})),
          b + ": M * I is the hull of the image");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        sums_and_translations(b);
        numbers_scale_and_a_negative_one_swaps_the_bounds(b);
        a_matrix_maps_the_box(b);
    });
    return test::finish("interval operators");
}
