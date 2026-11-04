// test_interval_abs - interval abs: the image of a box under the absolute value
//
// CORA: contSet/@interval/abs.m
// PROPERTY: contSet/@interval/abs.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// The three cases of a bound pair: below zero, across zero, above zero.
void abs_of_the_three_cases(const std::string &b) {
    const Interval I(column({-3.0, -2.0, 1.0}), column({-1.0, 4.0, 2.0}));
    const Interval A = I.abs();
    check(close(A.inf(), std::vector<double>{1.0, 0.0, 1.0}), b + ": abs lower bounds");
    check(close(A.sup(), std::vector<double>{3.0, 4.0, 2.0}), b + ": abs upper bounds");
}

/// A batch of boxes gives the batch of their images; a matrix of intervals is elementwise.
void abs_of_a_batch_and_a_matrix(const std::string &b) {
    const Interval I1(column({-3.0, 1.0}), column({-1.0, 2.0})),
        I2(column({-1.0, -5.0}), column({2.0, 0.5}));
    const Interval A = Interval::stack({I1, I2}).abs();
    check(close(A.inf(), Interval::stack({I1.abs(), I2.abs()}).inf()) &&
              close(A.sup(), Interval::stack({I1.abs(), I2.abs()}).sup()),
          b + ": the batch is the batch of the images");
    const Interval M(Tensor({{-1.0, 0.0}, {2.0, -3.0}}), Tensor({{1.0, 4.0}, {2.0, -1.0}}));
    check(close(M.abs().sup(), std::vector<double>{1.0, 4.0, 2.0, 3.0}), b + ": matrix, upper");
}

/// Every point of the box maps into the result, and both bounds are reached.
void abs_encloses_the_image_exactly(const std::string &b) {
    cora::Rng rng(11);
    for (const int n : {1, 3, 6}) {
        const Interval I = Interval::generate_random(rng, n);
        const Interval A = I.abs();
        const Tensor x = I.rand_point(40, rng); // (n, 40)
        bool inside = true;
        for (int k = 0; k < 40; ++k) inside = inside && A.contains(x.abs().slice(-1, k, k + 1));
        check(inside, b + ": a sample maps into abs");
        const Tensor corners = I.inf().abs().maximum(I.sup().abs());
        check(close(A.sup(), corners), b + ": the largest |bound| is the upper bound");
    }
}

/// abs has no input to refuse; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval S(column({-1, -2}), column({1, 3}));
    const Tensor on = S.abs().sup();
    ChecksScope off(false);
    check(close(S.abs().sup(), on, 0), "abs: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        abs_of_the_three_cases(b);
        abs_of_a_batch_and_a_matrix(b);
        abs_encloses_the_image_exactly(b);
    });
    checks_and_their_switch();
    return test::finish("interval abs");
}
