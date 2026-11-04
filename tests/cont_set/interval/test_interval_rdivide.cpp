// test_interval_rdivide - interval rdivide: the entrywise quotient of boxes and numbers
//
// CORA: contSet/@interval/rdivide.m
// PROPERTY: contSet/@interval/rdivide.m

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

/// A box over numbers: the hull of the quotients, the whole line where the number is 0.
void box_over_numbers(const std::string &b) {
    const Interval I(column({2.0, -4.0, 1.0}), column({4.0, 6.0, 3.0}));
    const Interval R = I.rdivide(column({2.0, -2.0, 0.0}));
    check(R.inf().data() == std::vector<double>({1, -3, -kInf}) &&
              R.sup().data() == std::vector<double>({2, 2, kInf}),
          b + ": the hull of the quotients, the line over 0");
    const Interval S = I.rdivide(Tensor::from_data({2.0}, {1, 1}));
    check(close(S.sup(), std::vector<double>{2, 3, 1.5}), b + ": one number divides every entry");
    const Interval T = Interval::stack({I, I}).rdivide(column({2.0, -2.0, 0.0}));
    check(T.inf().shape() == std::vector<int64_t>({2, 3, 1}), b + ": a batch has a batch shape");
}

/// Numbers over a box: both ends, a half line at 0, the whole line around 0.
void numbers_over_a_box(const std::string &b) {
    const Interval D(column({2.0, -4.0, 0.0, -2.0, -1.0}), column({4.0, -2.0, 2.0, 0.0, 2.0}));
    const Interval R = Interval::rdivide(Tensor::from_data({1.0}, {1, 1}), D);
    check(R.inf().data() == std::vector<double>({0.25, -0.5, 0.5, -kInf, -kInf}) &&
              R.sup().data() == std::vector<double>({0.5, -0.25, kInf, -0.5, kInf}),
          b + ": 1 ./ D");
    // CORA gives [-Inf, n/inf] at sup = 0 for every n, the opposite ray for n < 0
    const Interval Q = Interval::rdivide(Tensor::from_data({-1.0}, {1, 1}), D);
    check(Q.inf().data() == std::vector<double>({-0.5, 0.25, -kInf, 0.5, -kInf}) &&
              Q.sup().data() == std::vector<double>({-0.25, 0.5, -0.5, kInf, kInf}),
          b + ": -1 ./ D, the ray at sup = 0 is the positive one");
    const Interval P = Interval::rdivide(column({2.0, 0.0, 6.0, 3.0, 1.0}), D);
    check(P.inf().data()[1] == 0 && P.sup().data()[1] == 0, b + ": 0 ./ D is 0");
    check(P.inf().data()[0] == 0.5 && P.sup().data()[0] == 1.0 && P.inf().data()[3] == -kInf &&
              P.sup().data()[3] == -1.5,
          b + ": numbers of the shape of D");
}

/// A box over a box is I .* (1 ./ D).
void box_over_box(const std::string &b) {
    const Interval I(column({1.0, -2.0}), column({2.0, 3.0})),
        D(column({2.0, -4.0}), column({4.0, -2.0}));
    const Interval R = I.rdivide(D);
    check(close(R.inf(), std::vector<double>{0.25, -1.5}) &&
              close(R.sup(), std::vector<double>{1, 1}),
          b + ": I ./ D");
    const Interval W = I.rdivide(Interval(column({-1.0, 1.0}), column({1.0, 2.0})));
    check(W.inf().data()[0] == -kInf && W.sup().data()[0] == kInf, b + ": D around 0");
}

/// The quotient of every sampled pair of points is in the result.
void rdivide_encloses_the_image(const std::string &b) {
    cora::Rng rng(52);
    for (int trial = 0; trial < 20; ++trial) {
        const Interval I = Interval::generate_random(rng, 2);
        const Interval D(column({0.5 + trial % 3, -3.0 - trial % 2}),
                         column({1.5 + trial % 3, -0.5}));
        const Interval R = I.rdivide(D);
        const Tensor x = I.rand_point(20, rng), y = D.rand_point(20, rng);
        bool inside = true;
        for (int k = 0; k < 20; ++k)
            inside = inside && R.contains(x.slice(-1, k, k + 1).divide(y.slice(-1, k, k + 1)));
        check(inside, b + ": x / y is in the result");
    }
}

/// Inputs of other sizes and a denominator [0, 0] or a NaN quotient are refused; with the checks
/// off the result is the same.
void checks_and_their_switch() {
    const Interval I(column({1.0, 2.0}), column({2.0, 3.0}));
    check(test::raises([&] { I.rdivide(column({1.0, 2.0, 3.0})); }, Error::special_error),
          "rdivide: three numbers for two coordinates");
    check(
        test::raises([&] { Interval::rdivide(column({1.0, 2.0, 3.0}), I); }, Error::special_error),
        "rdivide: three numerators for two coordinates");
    check(
        test::raises([&] { I.rdivide(Interval(column({1.0, 2.0, 3.0}), column({1.0, 2.0, 3.0}))); },
                     Error::special_error),
        "rdivide: three denominators for two coordinates");
    check(test::raises(
              [&] { Interval::rdivide(column({1.0}), Interval(column({0.0}), column({0.0}))); },
              Error::out_of_domain),
          "rdivide: the denominator [0, 0]");
    const Interval U(column({1.0}), column({kInf}));
    check(test::raises([&] { U.rdivide(column({kInf})); }, Error::out_of_domain),
          "rdivide: Inf / Inf");
    const Tensor on = I.rdivide(column({2.0, 4.0})).sup();
    ChecksScope off(false);
    check(close(I.rdivide(column({2.0, 4.0})).sup(), on, 0),
          "rdivide: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        box_over_numbers(b);
        numbers_over_a_box(b);
        box_over_box(b);
        rdivide_encloses_the_image(b);
    });
    checks_and_their_switch();
    return test::finish("interval rdivide");
}
