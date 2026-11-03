// test_interval_sqrt - interval sqrt: the square root of a box
//
// CORA: contSet/@interval/sqrt.m
// PROPERTY: contSet/@interval/sqrt.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <cmath>
#include <limits>

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

void sqrt_maps_the_bounds(const std::string &b) {
    const Interval I(column({0.0, 1.0}), column({4.0, kInf}));
    const Interval R = I.sqrt();
    check(close(R.inf(), std::vector<double>{0.0, 1.0}) && R.sup().data()[1] == kInf &&
              close(R.sup().slice(-2, 0, 1), std::vector<double>{2.0}),
          b + ": sqrt of the bounds");
    const Interval S = Interval::stack({I, I}).sqrt();
    check(S.inf().shape() == std::vector<int64_t>({2, 2, 1}), b + ": a batch has a batch shape");
}

/// The root of every sampled point lies in the result, which is tight at the bounds.
void sqrt_encloses_the_image(const std::string &b) {
    cora::Rng rng(81);
    for (int trial = 0; trial < 20; ++trial) {
        const Interval I = Interval::generate_random(3, rng).abs(), R = I.sqrt();
        const Tensor x = I.rand_point(20, rng);
        bool inside = true;
        for (int k = 0; k < 20; ++k) inside = inside && R.contains(x.slice(-1, k, k + 1).sqrt());
        check(inside, b + ": sqrt of a sample is in the result");
        check(close(R.inf(), I.inf().sqrt(), 0) && close(R.sup(), I.sup().sqrt(), 0),
              b + ": the bounds are the images of the bounds");
    }
}

/// A negative lower bound is out of the domain; with the checks off the result is the same inside it.
void checks_and_their_switch() {
    const Interval I(column({0.5, 1.0}), column({1.0, 2.0}));
    check(test::raises([] { Interval(column({-1.0}), column({1.0})).sqrt(); }, Error::out_of_domain),
          "sqrt: a negative lower bound");
    check(test::raises([] { Interval(column({-2.0}), column({-1.0})).sqrt(); }, Error::out_of_domain),
          "sqrt: a negative box");
    const Tensor on = I.sqrt().sup();
    ChecksScope off(false);
    check(close(I.sqrt().sup(), on, 0), "sqrt: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        sqrt_maps_the_bounds(b);
        sqrt_encloses_the_image(b);
    });
    checks_and_their_switch();
    return test::finish("interval sqrt");
}
