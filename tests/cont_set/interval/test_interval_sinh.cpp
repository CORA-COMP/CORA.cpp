// test_interval_sinh - interval sinh: the hyperbolic sine of a box
//
// CORA: contSet/@interval/sinh.m
// PROPERTY: contSet/@interval/sinh.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

void sinh_maps_the_bounds(const std::string &b) {
    const Interval I(column({-1.0, 0.0}), column({2.0, 0.5}));
    const Interval R = I.sinh();
    check(close(R.inf(), std::vector<double>{std::sinh(-1.0), 0.0}) &&
              close(R.sup(), std::vector<double>{std::sinh(2.0), std::sinh(0.5)}),
          b + ": sinh of the bounds");
    const Interval S = Interval::stack({I, I}).sinh();
    check(S.inf().shape() == std::vector<int64_t>({2, 2, 1}), b + ": a batch has a batch shape");
    check(close(S.sup().slice(0, 1, 2).reshape({2, 1}), R.sup()), b + ": a batch member");
}

/// The image of every sampled point lies in the result, which is tight at the bounds.
void sinh_encloses_the_image(const std::string &b) {
    cora::Rng rng(21);
    for (int trial = 0; trial < 20; ++trial) {
        const Interval I = Interval::generate_random(rng, 3), R = I.sinh();
        const Tensor x = I.rand_point(20, rng);
        bool inside = true;
        for (int k = 0; k < 20; ++k) inside = inside && R.contains(x.slice(-1, k, k + 1).sinh());
        check(inside, b + ": sinh of a sample is in the result");
        check(close(R.inf(), I.inf().sinh(), 0) && close(R.sup(), I.sup().sinh(), 0),
              b + ": the bounds are the images of the bounds");
    }
}

/// sinh has no input to refuse; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval I(column({-1.0, 0.0}), column({1.0, 1.0}));
    const Tensor on = I.sinh().sup();
    ChecksScope off(false);
    check(close(I.sinh().sup(), on, 0), "sinh: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        sinh_maps_the_bounds(b);
        sinh_encloses_the_image(b);
    });
    checks_and_their_switch();
    return test::finish("interval sinh");
}
