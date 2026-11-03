// test_interval_log - interval log: the natural logarithm of a box
//
// CORA: contSet/@interval/log.m
// PROPERTY: contSet/@interval/log.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <algorithm>
#include <cmath>
#include <limits>

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

void log_maps_the_bounds(const std::string &b) {
    const Interval I(column({1.0, 0.5}), column({std::exp(1.0), 8.0}));
    const Interval R = I.log();
    check(close(R.inf(), std::vector<double>{0.0, std::log(0.5)}) &&
              close(R.sup(), std::vector<double>{1.0, std::log(8.0)}),
          b + ": log of the bounds");
    const Interval Z = Interval(column({0.0}), column({1.0})).log();
    check(Z.inf().data()[0] == -kInf && Z.sup().data()[0] == 0.0, b + ": log(0) is -Inf");
    check(Interval(column({1.0}), column({kInf})).log().sup().data()[0] == kInf,
          b + ": log(Inf) is Inf");
    const Interval S = Interval::stack({I, I}).log();
    check(S.inf().shape() == std::vector<int64_t>({2, 2, 1}), b + ": a batch has a batch shape");
    check(close(S.sup().slice(0, 1, 2).reshape({2, 1}), R.sup()), b + ": a batch member");
}

/// The logarithm of every sampled point lies in the result, which is tight at the bounds.
void log_encloses_the_image(const std::string &b) {
    cora::Rng rng(31);
    for (int trial = 0; trial < 20; ++trial) {
        std::vector<double> lo(3), hi(3);
        rng.uniform(lo.data(), 3, 0.01, 5);
        rng.uniform(hi.data(), 3, 0.01, 5);
        for (int i = 0; i < 3; ++i)
            if (lo[i] > hi[i]) std::swap(lo[i], hi[i]);
        const Interval I(column(lo), column(hi)), R = I.log();
        const Tensor x = I.rand_point(20, rng);
        bool inside = true;
        for (int k = 0; k < 20; ++k) inside = inside && R.contains(x.slice(-1, k, k + 1).log());
        check(inside, b + ": log of a sample is in the result");
        check(close(R.inf(), I.inf().log(), 0) && close(R.sup(), I.sup().log(), 0),
              b + ": the bounds are the images of the bounds");
    }
}

/// A negative bound is out of the domain; with the checks off the result is the same inside it.
void checks_and_their_switch() {
    const Interval I(column({0.5, 1.0}), column({1.0, 2.0}));
    check(test::raises([] { Interval(column({-1.0}), column({1.0})).log(); }, Error::out_of_domain),
          "log: a negative lower bound");
    const Tensor on = I.log().sup();
    ChecksScope off(false);
    check(close(I.log().sup(), on, 0), "log: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        log_maps_the_bounds(b);
        log_encloses_the_image(b);
    });
    checks_and_their_switch();
    return test::finish("interval log");
}
