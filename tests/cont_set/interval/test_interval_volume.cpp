// test_interval_volume - interval volume: the volume of a box (CORA's volume_)
//
// CORA: contSet/@interval/volume_.m
// PROPERTY: contSet/@interval/volume_.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <limits>

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

void volume_is_the_product_of_the_widths(const std::string &b) {
    const Interval I(column({1.0, -2.0}), column({3.0, 4.0}));
    check(I.volume().shape() == std::vector<int64_t>({1, 1}) &&
              close(I.volume(), std::vector<double>{12}),
          b + ": 2 * 6");
    check(close(Interval(column({1.0, 1.0}), column({3.0, 1.0})).volume(), std::vector<double>{0}),
          b + ": a degenerate box has volume 0");
    check(close(Interval(column({0.0, 1.0}), column({kInf, 1.0})).volume(), std::vector<double>{0}),
          b + ": 0 next to an infinite width is 0");
    check(Interval(column({0.0, 1.0}), column({kInf, 2.0})).volume().data()[0] == kInf,
          b + ": an unbounded box has infinite volume");
    check(close(Interval::empty(3).volume(), std::vector<double>{0}), b + ": the empty box");
    const Tensor lo = Tensor::from_data({0, 0, 0, 0}, {2, 2});
    check(close(Interval(lo, lo + 2.0).volume(), std::vector<double>{16}),
          b + ": an interval matrix multiplies all entries");
    const Interval S = Interval::stack({I, Interval(column({0.0, 0.0}), column({1.0, 2.0}))});
    check(S.volume().shape() == std::vector<int64_t>({2, 1, 1}) &&
              close(S.volume(), std::vector<double>{12, 2}),
          b + ": a batch has a volume per member");
}

/// The volume scales with the product of the factors, and the center does not matter.
void volume_scales(const std::string &b) {
    cora::Rng rng(61);
    for (int trial = 0; trial < 10; ++trial) {
        const Interval I = Interval::generate_random(3, rng);
        const Tensor v = I.volume();
        check(close(v, (I.rad() * 2.0).prod(-2, true), 1e-12), b + ": 2 rad multiplied");
        check(close((I + column({1.0, -2.0, 3.0})).volume(), v, 1e-12), b + ": a shift keeps it");
        check(close((2.0 * I).volume(), v * 8.0, 1e-12), b + ": scaling by 2 in 3 dimensions");
    }
}

/// volume has no input to refuse; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval I(column({1.0, -2.0}), column({3.0, 4.0}));
    const Tensor on = I.volume();
    ChecksScope off(false);
    check(close(I.volume(), on, 0), "volume: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        volume_is_the_product_of_the_widths(b);
        volume_scales(b);
    });
    checks_and_their_switch();
    return test::finish("interval volume");
}
