// test_interval_times - interval times: the entrywise product with a box or with numbers
//
// CORA: contSet/@interval/times.m
// PROPERTY: contSet/@interval/times.m

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

void times_takes_the_hull_of_the_corners(const std::string &b) {
    const Interval A(column({1.0, -2.0}), column({3.0, 4.0})), B(column({2.0, 0.0}), column({5.0, 1.0}));
    const Interval R = A.times(B);
    check(close(R.inf(), std::vector<double>{2, -2}) && close(R.sup(), std::vector<double>{15, 4}),
          b + ": box times box");
    const Interval N = A.times(column({-2.0, 3.0}));
    check(close(N.inf(), std::vector<double>{-6, -6}) && close(N.sup(), std::vector<double>{-2, 12}),
          b + ": box times numbers");
    const Interval Z = A.times(Tensor::from_data({0.0}, {1, 1}));
    check(close(Z.inf(), std::vector<double>{0, 0}) && close(Z.sup(), std::vector<double>{0, 0}),
          b + ": a scalar 0");
    const Interval S = A.times(Tensor::from_data({-1.0}, {1, 1}));
    check(close(S.inf(), std::vector<double>{-3, -4}) && close(S.sup(), std::vector<double>{-1, 2}),
          b + ": a negative scalar swaps the bounds");
    const Interval U(column({0.0}), column({kInf}));
    check(close(U.times(Interval(column({0.0}), column({0.0}))).sup(), std::vector<double>{0}),
          b + ": 0 * Inf is 0");
    const Interval W = A.times(Interval(column({2.0}), column({3.0})));
    check(W.inf().shape() == std::vector<int64_t>({2, 1}) &&
              close(W.sup(), std::vector<double>{9, 12}),
          b + ": a scalar box broadcasts");
    const Interval Bt = Interval::stack({A, A}).times(B);
    check(Bt.inf().shape() == std::vector<int64_t>({2, 2, 1}), b + ": a batch has a batch shape");
}

/// The product of sampled points lies in the result; it commutes and the box times itself matches
/// the numbers when the other is a point.
void times_encloses_the_products(const std::string &b) {
    cora::Rng rng(111);
    for (int trial = 0; trial < 10; ++trial) {
        const Interval A = Interval::generate_random(3, rng), B = Interval::generate_random(3, rng);
        const Interval R = A.times(B);
        const Tensor x = A.rand_point(20, rng), y = B.rand_point(20, rng);
        bool inside = true;
        for (int k = 0; k < 20; ++k)
            inside = inside && R.contains(x.slice(-1, k, k + 1).multiply(y.slice(-1, k, k + 1)));
        check(inside, b + ": the product of samples is in the result");
        check(close(R.inf(), B.times(A).inf(), 0) && close(R.sup(), B.times(A).sup(), 0),
              b + ": times commutes");
        const Tensor p = B.center();
        check(close(A.times(p).inf(), A.times(Interval(p, p)).inf(), 0) &&
                  close(A.times(p).sup(), A.times(Interval(p, p)).sup(), 0),
              b + ": numbers are a point box");
    }
}

/// The shapes must broadcast; with the checks off a valid call gives the same result.
void checks_and_their_switch() {
    const Interval A(column({1.0, -2.0}), column({3.0, 4.0}));
    const Interval C(column({1.0, 2.0, 3.0}), column({2.0, 3.0, 4.0}));
    check(test::raises([&] { A.times(C); }, Error::dimension_mismatch), "times: boxes of 2 and 3");
    check(test::raises([&] { A.times(column({1.0, 2.0, 3.0})); }, Error::dimension_mismatch),
          "times: numbers of 3");
    const Tensor on = A.times(A).sup();
    ChecksScope off(false);
    check(close(A.times(A).sup(), on, 0), "times: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        times_takes_the_hull_of_the_corners(b);
        times_encloses_the_products(b);
    });
    checks_and_their_switch();
    return test::finish("interval times");
}
