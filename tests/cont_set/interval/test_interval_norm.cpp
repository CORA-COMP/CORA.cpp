// test_interval_norm - interval norm: the norm of the largest member of a box
//
// CORA: contSet/@interval/norm_.m

#include "cora/cont_set/interval/interval.h"
#include "testing.h"

#include <cmath>
#include <limits>

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

void norm_of_the_largest_member(const std::string &b) {
    // max(|inf|, |sup|) = (3, 4); the center is (-1, -1) and the radius (2, 3)
    const Interval I(column({-3.0, -4.0}), column({1.0, 2.0}));
    check(close(I.norm(), std::vector<double>{5.0}), b + ": the 2-norm of max(|inf|, |sup|)");
    check(close(I.norm(Tensor::Norm::one), std::vector<double>{7.0}),
          b + ": the 1-norm of |c| + r");
    check(close(I.norm(Tensor::Norm::inf), std::vector<double>{4.0}), b + ": the inf-norm");
    check(close(I.norm(Tensor::Norm::fro), std::vector<double>{5.0}), b + ": the Frobenius norm");
    const Interval M(Tensor::from_data({-3, 0, 0, -4}, {2, 2}),
                     Tensor::from_data({1, 0, 0, 2}, {2, 2}));
    check(close(M.norm(), std::vector<double>{4.0}), b + ": a matrix has the spectral norm");
    check(Interval::empty(2).norm().data()[0] == -std::numeric_limits<double>::infinity(),
          b + ": the empty interval has norm -Inf");
    const Tensor N = Interval::stack({I, I}).norm();
    check(N.shape() == std::vector<int64_t>({2, 1, 1}) &&
              close(N.reshape({2}), std::vector<double>{5, 5}),
          b + ": a batch");
}

/// norm has no input to refuse; with the checks off the result is the same.
void checks_and_their_switch() {
    const Interval I(column({-3.0, -4.0}), column({1.0, 2.0}));
    ChecksScope off(false);
    check(close(I.norm(), std::vector<double>{5.0}), "norm: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(norm_of_the_largest_member);
    checks_and_their_switch();
    return test::finish("interval norm");
}
