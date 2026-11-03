// test_interval_tan - interval tan: the tangent of a box
//
// CORA: contSet/@interval/tan.m
// PROPERTY: contSet/@interval/tan.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <cmath>
#include <limits>
#include <numbers>

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

void tan_maps_the_bounds_or_gives_the_line(const std::string &b) {
    const Interval I(column({-1.0, 1.0, 1.5, 0.0}), column({1.0, 3.0, 4.6, 4.0}));
    const Interval R = I.tan();
    check(close(R.inf().slice(-2, 0, 1), std::vector<double>{std::tan(-1.0)}) &&
              close(R.sup().slice(-2, 0, 1), std::vector<double>{std::tan(1.0)}),
          b + ": tan of the bounds without a pole");
    const std::vector<double> lo = R.inf().data(), hi = R.sup().data();
    check(lo[1] == -kInf && hi[1] == kInf, b + ": a pole at pi / 2 inside (width < pi)");
    check(lo[2] == -kInf && hi[2] == kInf, b + ": a pole inside, tan(inf) > tan(sup)");
    check(lo[3] == -kInf && hi[3] == kInf, b + ": a width of at least pi");
    const Interval S = Interval::stack({I, I}).tan();
    check(S.inf().shape() == std::vector<int64_t>({2, 4, 1}), b + ": a batch has a batch shape");
    check(S.sup().slice(0, 1, 2).reshape({4, 1}).data() == R.sup().data(), b + ": a batch member");
}

/// The tangent of every sampled point lies in the result: a box without a pole is mapped by
/// the bounds, a box with one gives the whole line.
void tan_encloses_the_image(const std::string &b) {
    cora::Rng rng(101);
    for (int trial = 0; trial < 20; ++trial) {
        std::vector<double> lo(3), hi(3);
        rng.uniform(lo.data(), 3, -4, 4);
        rng.uniform(hi.data(), 3, 0, 2);
        for (int i = 0; i < 3; ++i) hi[i] += lo[i];
        const Interval I(column(lo), column(hi)), R = I.tan();
        const Tensor x = I.rand_point(20, rng);
        bool inside = true;
        for (int k = 0; k < 20; ++k) inside = inside && R.contains(x.slice(-1, k, k + 1).tan());
        check(inside, b + ": tan of a sample is in the result");
    }
}

/// tan has no input to refuse; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval I(column({-1.0, 0.0}), column({1.0, 1.0}));
    const Tensor on = I.tan().sup();
    ChecksScope off(false);
    check(close(I.tan().sup(), on, 0), "tan: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        tan_maps_the_bounds_or_gives_the_line(b);
        tan_encloses_the_image(b);
    });
    checks_and_their_switch();
    return test::finish("interval tan");
}
